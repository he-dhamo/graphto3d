from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset.dataset import RIODatasetSceneGraph, collate_fn_vaegan_points
from model.VAE import VAE
from model.atlasnet import AE_AtlasNet
from model.discriminators import BoxDiscriminator, ShapeAuxillary
from model.losses import bce_loss
from helpers.util import bool_flag

from model.losses import calculate_model_losses

import torch.nn.functional as F
import json

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
# standard hyperparameters, batch size, learning rate, etc
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--nepoch', type=int, default=101, help='number of epochs to train for')

# paths and filenames
parser.add_argument('--outf', type=str, default='checkpoint', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', required=False, type=str, default="./GT", help="dataset path")
parser.add_argument('--dataset_3RScan', type=str, default='', help="dataset path of 3RScan")
parser.add_argument('--label_file', required=False, type=str, default='labels.instances.align.annotated.ply', help="label file name")
parser.add_argument('--logf', default='logs', help='folder to save tensorboard logs')
parser.add_argument('--exp', default='./experiments/layout_test', help='experiment name')
parser.add_argument('--path2atlas', default="./experiments/model_36.pth", type=str)

# GCN parameters
parser.add_argument('--residual', type=bool_flag, default=False, help="residual in GCN")
parser.add_argument('--pooling', type=str, default='avg', help="pooling method in GCN")

# dataset related
parser.add_argument('--large', default=True, type=bool_flag, help='large set of shape class labels')
parser.add_argument('--use_splits', default=True, type=bool_flag, help='Set to true if you want to use splitted training data')
parser.add_argument('--use_scene_rels', type=bool_flag, default=True, help="connect all nodes to a root scene node")
parser.add_argument('--with_points', type=bool_flag, default=False, help="if false and with_feats is false, only predicts layout."
                                                                         "If true always also loads pointsets. Notice that this is much "
                                                                         "slower than only loading the save features. Therefore, "
                                                                         "the recommended setting is with_points=False and with_feats=True.")
parser.add_argument('--with_feats', type=bool_flag, default=True, help="if true reads latent point features instead of pointsets."
                                                                       "If not existing, they get generated at the beginning.")
parser.add_argument('--shuffle_objs', type=bool_flag, default=True, help="shuffle objs of a scene")
parser.add_argument('--num_points', type=int, default=1024, help='number of points for each object')
parser.add_argument('--rio27', default=False, type=bool_flag)
parser.add_argument('--use_canonical', default=True, type=bool_flag)
parser.add_argument('--with_angles', default=True, type=bool_flag)
parser.add_argument('--num_box_params', default=6, type=int)
parser.add_argument('--crop_floor', default=False, type=bool_flag)

# training and architecture related
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--overfiting_debug', type=bool_flag, default=False)
parser.add_argument('--weight_D_box', default=0.1, type=float, help="Box Discriminator")
parser.add_argument('--with_changes', default=True, type=bool_flag)
parser.add_argument('--with_shape_disc', default=True, type=bool_flag)
parser.add_argument('--with_manipulator', default=True, type=bool_flag)

parser.add_argument('--replace_latent', default=True, type=bool_flag)
parser.add_argument('--network_type', default='shared', choices=['dis', 'sln', 'mlp', 'shared'], type=str)

args = parser.parse_args()
print(args)


def train():
    """ Train the network based on the provided argparse parameters
    """
    args.manualSeed = random.randint(1, 10000)  # optionally fix seed 7494
    print("Random Seed: ", args.manualSeed)

    print(torch.__version__)

    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # prepare pretrained AtlasNet model, later used to convert pointsets to a shape feature
    saved_atlasnet_model = torch.load(args.path2atlas)
    point_ae = AE_AtlasNet(num_points=1024, bottleneck_size=128, nb_primitives=25)
    point_ae.load_state_dict(saved_atlasnet_model, strict=True)
    if torch.cuda.is_available():
        point_ae = point_ae.cuda()
    point_ae.eval()

    # instantiate scene graph dataset for training
    dataset = RIODatasetSceneGraph(
            root=args.dataset,
            root_3rscan=args.dataset_3RScan,
            label_file=args.label_file,
            npoints=args.num_points,
            path2atlas=args.path2atlas,
            split='train_scans',
            shuffle_objs=(args.shuffle_objs and not args.overfiting_debug),
            use_points=args.with_points,
            use_scene_rels=args.use_scene_rels,
            with_changes=args.with_changes,
            vae_baseline=args.network_type == 'sln',
            with_feats=args.with_feats,
            large=args.large,
            atlas=point_ae,
            seed=False,
            use_splits=args.use_splits,
            use_rio27=args.rio27,
            use_canonical=args.use_canonical,
            crop_floor=args.crop_floor,
            center_scene_to_floor=args.crop_floor,
            recompute_feats=False)

    collate_fn = collate_fn_vaegan_points
    # instantiate data loader from dataset
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batchSize,
            collate_fn=collate_fn,
            shuffle=(not args.overfiting_debug),
            num_workers=int(args.workers))

    # number of object classes and relationship classes
    num_classes = len(dataset.classes)
    num_relationships = len(dataset.relationships)

    try:
        os.makedirs(args.outf)
    except OSError:
        pass
    # instantiate the model
    model = VAE(type=args.network_type, vocab=dataset.vocab, replace_latent=args.replace_latent,
                with_changes=args.with_changes, residual=args.residual, gconv_pooling=args.pooling,
                with_angles=args.with_angles, num_box_params=args.num_box_params)
    if torch.cuda.is_available():
        model = model.cuda()
    # instantiate a relationship discriminator that considers the boxes and the semantic labels
    # if the loss weight is larger than zero
    # also create an optimizer for it
    if args.weight_D_box > 0:
        boxD = BoxDiscriminator(6, num_relationships, num_classes)
        optimizerDbox = optim.Adam(filter(lambda p: p.requires_grad, boxD.parameters()), lr=args.lr, betas=(0.9, 0.999))
        boxD.cuda()
        boxD = boxD.train()

    # instantiate auxiliary discriminator for shape and a respective optimizer
    shapeClassifier = ShapeAuxillary(128, len(dataset.cat))
    shapeClassifier = shapeClassifier.cuda()
    shapeClassifier.train()
    optimizerShapeAux = optim.Adam(filter(lambda p: p.requires_grad, shapeClassifier.parameters()), lr=args.lr, betas=(0.9, 0.999))

    # initialize tensorboard writer
    writer = SummaryWriter(args.exp + "/" + args.logf)

    # optimizer for model
    params = filter(lambda p: p.requires_grad,list(model.parameters()) )
    optimizer = optim.Adam(params, lr=args.lr)

    print("---- Model and Dataset built ----")

    if not os.path.exists(args.exp + "/" + args.outf):
        os.makedirs(args.exp + "/" + args.outf)

    # save parameters so that we can read them later on evaluation
    with open(os.path.join(args.exp, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print("Saving all parameters under:")
    print(os.path.join(args.exp, 'args.json'))

    optimizer.step()
    torch.autograd.set_detect_anomaly(True)
    counter = 0

    print("---- Starting training loop! ----")
    for epoch in range(0, args.nepoch):
        print('Epoch: {}/{}'.format(epoch, args.nepoch))

        for i, data in enumerate(dataloader, 0):
            # skip invalid data
            if data == -1:
                continue

            try:
                enc_objs, enc_triples, enc_tight_boxes, enc_objs_to_scene, enc_triples_to_scene = data['encoder']['objs'],\
                            data['encoder']['tripltes'], data['encoder']['boxes'], data['encoder']['obj_to_scene'], data['encoder']['tiple_to_scene']

                if args.with_feats:
                    encoded_enc_points = data['encoder']['feats']
                    encoded_enc_points = encoded_enc_points.cuda()
                elif args.with_points:
                    enc_points = data['encoder']['points']
                    enc_points = enc_points.cuda()

                dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'],\
                            data['decoder']['tripltes'], data['decoder']['boxes'], data['decoder']['obj_to_scene'], data['decoder']['tiple_to_scene']

                if 'feats' in data['decoder']:
                    encoded_dec_points = data['decoder']['feats']
                    encoded_dec_points = encoded_dec_points.cuda()
                else:
                    if 'points' in data['decoder']:
                        dec_points = data['decoder']['points']
                        dec_points = dec_points.cuda()

                # changed nodes
                missing_nodes = data['missing_nodes']
                manipulated_nodes = data['manipulated_nodes']

            except Exception as e:
                print('Exception', str(e))
                continue

            enc_objs, enc_triples, enc_tight_boxes = enc_objs.cuda(), enc_triples.cuda(), enc_tight_boxes.cuda()
            dec_objs, dec_triples, dec_tight_boxes = dec_objs.cuda(), dec_triples.cuda(), dec_tight_boxes.cuda()

            if args.with_points:
                enc_points, dec_points = enc_points.cuda(), dec_points.cuda()

            # avoid batches with insufficient number of instances with valid shape classes
            mask = [ob in dataset.point_classes_idx for ob in dec_objs]
            if sum(mask) <= 1:
                continue

            optimizer.zero_grad()
            optimizerShapeAux.zero_grad()

            model = model.train()

            if args.weight_D_box > 0:
                optimizerDbox.zero_grad()

            # if we are reading pointsets and not precomputed shape features, convert them to features
            # otherwise do nothing, as we already have the features
            if not args.with_feats and args.with_points:
                with torch.no_grad():
                    encoded_enc_points = point_ae.encoder(enc_points.transpose(2,1).contiguous())
                    encoded_dec_points = point_ae.encoder(dec_points.transpose(2,1).contiguous())

            # set all scene (dummy) nodes point encodings to zero
            enc_scene_nodes = enc_objs == 0
            dec_scene_nodes = dec_objs == 0
            encoded_enc_points[enc_scene_nodes] = torch.zeros([torch.sum(enc_scene_nodes), encoded_enc_points.shape[1]]).float().cuda()
            encoded_dec_points[dec_scene_nodes] = torch.zeros([torch.sum(dec_scene_nodes), encoded_dec_points.shape[1]]).float().cuda()

            if args.num_box_params == 7:
                # all parameters, including angle, procesed by the box_net
                enc_boxes = enc_tight_boxes
                dec_boxes = dec_tight_boxes
            elif args.num_box_params == 6:
                # no angle. this will be learned separately if with_angle is true
                enc_boxes = enc_tight_boxes[:, :6]
                dec_boxes = dec_tight_boxes[:, :6]
            elif args.num_box_params == 4:
                # height, centroid. assuming we want the other sizes to be estimated from the shape aspect ratio
                enc_boxes = enc_tight_boxes[:, 2:6]
                dec_boxes = dec_tight_boxes[:, 2:6]
            else:
                raise NotImplementedError

            # limit the angle bin range from 0 to 24
            enc_angles = enc_tight_boxes[:, 6].long() - 1
            enc_angles = torch.where(enc_angles > 0, enc_angles, torch.zeros_like(enc_angles))
            enc_angles = torch.where(enc_angles < 24, enc_angles, torch.zeros_like(enc_angles))
            dec_angles = dec_tight_boxes[:, 6].long() - 1
            dec_angles = torch.where(dec_angles > 0, dec_angles, torch.zeros_like(dec_angles))
            dec_angles = torch.where(dec_angles < 24, dec_angles, torch.zeros_like(dec_angles))

            attributes = None

            boxGloss = 0
            loss_genShape = 0
            loss_genShapeFake = 0
            loss_shape_fake_g = 0

            if args.with_manipulator:
                model_out = model.forward_mani(enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_points, attributes, enc_objs_to_scene,
                                                   dec_objs, dec_triples, dec_boxes, dec_angles, encoded_dec_points, attributes, dec_objs_to_scene,
                                                   missing_nodes, manipulated_nodes)

                mu_box, logvar_box, mu_shape, logvar_shape, orig_gt_box, orig_gt_angle, orig_gt_shape, orig_box, orig_angle, orig_shape, \
                dec_man_enc_box_pred, dec_man_enc_angle_pred, dec_man_enc_shape_pred, keep = model_out
            else:
                model_out = model.forward_no_mani(dec_objs, dec_triples, dec_boxes, encoded_dec_points, angles=dec_angles,
                                      attributes=attributes)

                mu_box, logvar_box, mu_shape, logvar_shape, dec_man_enc_box_pred, dec_man_encd_angles_pred, \
                  dec_man_enc_shape_pred = model_out

                orig_gt_box = dec_boxes
                orig_box = dec_man_enc_box_pred

                orig_gt_shape = encoded_dec_points
                orig_shape = dec_man_enc_shape_pred

                orig_angle = dec_man_encd_angles_pred
                orig_gt_angle = dec_angles

                keep = []
                for i in range(len(dec_man_enc_box_pred)):
                    keep.append(1)
                keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

            if args.with_manipulator and args.with_shape_disc and dec_man_enc_shape_pred is not None:
                shape_logits_fake_d, probs_fake_d = shapeClassifier(dec_man_enc_shape_pred[mask].detach())
                shape_logits_fake_g, probs_fake_g = shapeClassifier(dec_man_enc_shape_pred[mask])
                shape_logits_real, probs_real = shapeClassifier(encoded_dec_points[mask].detach())

                # auxiliary loss. can the discriminator predict the correct class for the generated shape?
                loss_shape_real = torch.nn.functional.cross_entropy(shape_logits_real, dec_objs[mask])
                loss_shape_fake_d = torch.nn.functional.cross_entropy(shape_logits_fake_d, dec_objs[mask])
                loss_shape_fake_g = torch.nn.functional.cross_entropy(shape_logits_fake_g, dec_objs[mask])
                # standard discriminator loss
                loss_genShapeFake = bce_loss(probs_fake_g, torch.ones_like(probs_fake_g))
                loss_dShapereal = bce_loss(probs_real, torch.ones_like(probs_real))
                loss_dShapefake = bce_loss(probs_fake_d, torch.zeros_like(probs_fake_d))

                loss_dShape = loss_dShapefake + loss_dShapereal + loss_shape_real + loss_shape_fake_d
                loss_genShape = loss_genShapeFake + loss_shape_fake_g
                loss_dShape.backward()
                optimizerShapeAux.step()

            vae_loss_box, vae_losses_box = calculate_model_losses(args,
                                                                    orig_gt_box,
                                                                    orig_box,
                                                                    name='box', withangles=args.with_angles, angles_pred=orig_angle,
                                                                    mu=mu_box, logvar=logvar_box, angles=orig_gt_angle,
                                                                    KL_weight=0.1, writer=writer, counter=counter)
            if dec_man_enc_shape_pred is not None:
                vae_loss_shape, vae_losses_shape = calculate_model_losses(args,
                                                                        orig_gt_shape,
                                                                        orig_shape,
                                                                        name='shape', withangles=False,
                                                                        mu=mu_shape, logvar=logvar_shape,
                                                                        KL_weight=0.1, writer=writer, counter=counter)
            else:
                # set shape loss to 0 if we are only predicting layout
                vae_loss_shape, vae_losses_shape = 0, 0

            if args.with_manipulator and args.with_changes:
                oriented_gt_boxes = torch.cat([dec_boxes], dim=1)
                boxes_pred_in = keep * oriented_gt_boxes + (1-keep) * dec_man_enc_box_pred

                if args.weight_D_box == 0:
                    # Generator loss
                    boxGloss = 0
                    # Discriminator loss
                    gamma = 0.1
                    boxDloss_real = 0
                    boxDloss_fake = 0
                    reg_loss = 0
                else:
                    logits, _ = boxD(dec_objs, dec_triples, boxes_pred_in, keep)
                    logits_fake, reg_fake = boxD(dec_objs, dec_triples, boxes_pred_in.detach(), keep, with_grad=True,
                                               is_real=False)
                    logits_real, reg_real = boxD(dec_objs, dec_triples, oriented_gt_boxes, with_grad=True, is_real=True)
                    # Generator loss
                    boxGloss = bce_loss(logits, torch.ones_like(logits))
                    # Discriminator loss
                    gamma = 0.1
                    boxDloss_real = bce_loss(logits_real, torch.ones_like(logits_real))
                    boxDloss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake))
                    # Regularization by gradient penalty
                    reg_loss = torch.mean(reg_real + reg_fake)

                # gradient penalty
                # disc_reg = discriminator_regularizer(logits_real, in_real, logits_fake, in_fake)
                boxDloss = boxDloss_fake + boxDloss_real + (gamma/2.0) * reg_loss
                optimizerDbox.zero_grad()
                boxDloss.backward()
                # gradient clip
                # torch.nn.utils.clip_grad_norm_(boxD.parameters(), 5.0)
                optimizerDbox.step()

            loss = vae_loss_box + vae_loss_shape + 0.1 * loss_genShape
            if args.with_changes:
                   loss = loss + args.weight_D_box * boxGloss #+ b_loss

            # optimize
            loss.backward()

            # Cap the occasional super mutant gradient spikes
            # Do now a gradient step and plot the losses
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None and p.requires_grad and torch.isnan(p.grad).any():
                        print('NaN grad in step {}.'.format(counter))
                        p.grad[torch.isnan(p.grad)] = 0
            optimizer.step()
            counter += 1

            if counter % 100 == 0:
                print("loss at {}: box {:.4f}\tshape {:.4f}\tdiscr RealFake {:.4f}\t discr Classifcation "
                      "{:.4f}".format(counter, vae_loss_box, vae_loss_shape, loss_genShapeFake,
                                                              loss_shape_fake_g))
            writer.add_scalar('Train Loss BBox', vae_loss_box, counter)
            writer.add_scalar('Train Loss Shape', vae_loss_shape, counter)
            writer.add_scalar('Train Loss loss_genShapeFake', loss_genShapeFake, counter)
            writer.add_scalar('Train Loss loss_shape_fake_g', loss_shape_fake_g, counter)

        if epoch % 5 == 0:
            model.save(args.exp, args.outf, epoch)

    writer.close()


def main():
    train()


if __name__ == "__main__": main()
