import torch
import torch.nn as nn

import pickle
import os

from model.VAEGAN_DIS import Sg2ScVAEModel as Dis
from model.VAEGAN_SLN import Sg2ScVAEModel as SLN
from model.VAEGAN_SHARED import Sg2ScVAEModel as Shared
from model.shapeMlp import ShapeMLP


class VAE(nn.Module):

    def __init__(self, type='dis', vocab=None, replace_latent=False, with_changes=True, distribution_before=True,
                 residual=False, gconv_pooling='avg', with_angles=False, num_box_params=6):
        super().__init__()
        assert type in ['dis', 'sln', 'shared', 'mlp'], '{} is not included in [dis, sln, shared, mlp]'.format(type)

        self.type_ = type
        self.vocab = vocab
        self.with_angles = with_angles

        if self.type_ == 'dis':
            assert replace_latent is not None
            self.vae_box = Dis(vocab, embedding_dim=64, decoder_cat=True, mlp_normalization="batch",
                               input_dim=num_box_params, replace_latent=replace_latent, use_angles=with_angles,
                               residual=residual, gconv_pooling=gconv_pooling, gconv_num_layers=5)
            self.vae_shape = Dis(vocab, embedding_dim=128, decoder_cat=True, mlp_normalization="batch",
                                 input_dim=128, gconv_num_layers=5
                                 , replace_latent=replace_latent,
                                 residual=residual, gconv_pooling=gconv_pooling, use_angles=False)

        elif self.type_ == 'shared':
            assert distribution_before is not None and replace_latent is not None and with_changes is not None
            self.vae = Shared(vocab, embedding_dim=128, decoder_cat=True, mlp_normalization="batch",
                              gconv_num_layers=5, gconv_num_shared_layer=5, with_changes=with_changes, use_angles=with_angles,
                              distribution_before=distribution_before, replace_latent=replace_latent,
                              num_box_params=num_box_params, residual=residual)

        elif self.type_ == 'mlp':
            self.vae_box = Dis(vocab, embedding_dim=64, decoder_cat=True, mlp_normalization="batch",
                               input_dim=num_box_params, use_angles=with_angles)
            self.vae_shape = ShapeMLP(num_objs=len(vocab['object_idx_to_name']), embedding_dim=128)

        elif self.type_ == 'sln':
            self.vae_box = SLN(vocab, embedding_dim=64, decoder_cat=True, mlp_normalization="batch", use_attr=False,
                               Nangle=24)

    def forward_mani(self, enc_objs, enc_triples, enc_boxes, enc_angles, enc_shapes, attributes, enc_objs_to_scene, dec_objs,
                     dec_triples, dec_boxes, dec_angles, dec_shapes, dec_attributes, dec_objs_to_scene, missing_nodes,
                     manipulated_nodes):

        if self.type_ == 'shared':
            mu, logvar, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, orig_shapes, boxes, angles, shapes, keep = \
                self.vae.forward(enc_objs, enc_triples, enc_boxes, enc_angles, enc_shapes, attributes, enc_objs_to_scene,
                                 dec_objs, dec_triples, dec_boxes, dec_angles, dec_shapes, dec_attributes, dec_objs_to_scene,
                                 missing_nodes, manipulated_nodes)

            return mu, logvar, mu, logvar, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, orig_angles, orig_shapes, boxes, angles, shapes, keep

        elif self.type_ == 'dis':
            mu_boxes, logvar_boxes, orig_gt_boxes, orig_gt_angles, orig_boxes, orig_angles, boxes, angles, keep = self.vae_box.forward(
                enc_objs, enc_triples, enc_boxes, attributes, enc_objs_to_scene, dec_objs, dec_triples, dec_boxes,
                dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes, enc_angles, dec_angles)
            mu_shapes, logvar_shapes, orig_gt_shapes, _, orig_shapes, _, shapes, _, _ = self.vae_shape.forward(
                enc_objs, enc_triples, enc_shapes, attributes, enc_objs_to_scene, dec_objs, dec_triples, dec_shapes,
                dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes)

            return mu_boxes, logvar_boxes, mu_shapes, logvar_shapes, orig_gt_boxes, orig_gt_angles, orig_gt_shapes, orig_boxes, \
                   orig_angles, orig_shapes, boxes, angles, shapes, keep

        elif self.type_ == 'mlp':
            mu_boxes, logvar_boxes, orig_gt_boxes, orig_gt_angles, orig_boxes, orig_angles, boxes, angles, keep = self.vae_box.forward(
                enc_objs, enc_triples, enc_boxes, attributes, enc_objs_to_scene, dec_objs, dec_triples, dec_boxes,
                dec_attributes, dec_objs_to_scene, missing_nodes, manipulated_nodes, enc_angles, dec_angles)
            mu_shapes, logvar_shapes, shapes, _ = self.vae_shape.forward(dec_objs, dec_shapes)

            return mu_boxes, logvar_boxes, mu_shapes, logvar_shapes, orig_gt_boxes, orig_gt_angles, dec_shapes, orig_boxes, \
                   orig_angles, shapes, boxes, angles, shapes, keep

        elif self.type_ == 'sln':
            # sln baseline does not have a manipulation mode
            return None, None, None, None, None, None, None, None, None, None, None

    def forward_no_mani(self, objs, triples, boxes, shapes, angles=None, attributes=None):

        (mu_boxes, logvar_boxes), (mu_shapes, logvar_shapes) = self.encode_box_and_shape(objs, triples, shapes, boxes,
                                                                                angles=angles, attributes=attributes)
        # reparameterization
        std_box = torch.exp(0.5 * logvar_boxes)
        # standard sampling
        eps_box = torch.randn_like(std_box)

        z_boxes = eps_box.mul(std_box).add_(mu_boxes)
        z_shapes = None
        if mu_shapes is not None:
            std_shapes = torch.exp(0.5 * logvar_shapes)
            eps_shapes = torch.randn_like(std_shapes)
            z_shapes = eps_shapes.mul(std_shapes).add_(mu_shapes)

        boxes, angles, shapes = self.decoder_boxes_and_shape(z_boxes, z_shapes, objs, triples, attributes, None)
        return mu_boxes, logvar_boxes, mu_shapes, logvar_shapes, boxes, angles, shapes

    def load_networks(self, exp, epoch, strict=True):
        if self.type_ == 'dis':
            self.vae_box.load_state_dict(
                torch.load(os.path.join(exp, 'checkpoint', 'model_box_{}.pth'.format(epoch))),
                strict=strict
            )
            self.vae_shape.load_state_dict(
                torch.load(os.path.join(exp, 'checkpoint', 'model_shape_{}.pth'.format(epoch))),
                strict=strict
            )
        elif self.type_ == 'shared':
            print()
            ckpt = torch.load(os.path.join(exp, 'checkpoint', 'model{}.pth'.format(epoch))).state_dict()
            self.vae.load_state_dict(
                ckpt,
                strict=strict
            )
        elif self.type_ == 'mlp':
            self.vae_box.load_state_dict(
                torch.load(os.path.join(exp, 'checkpoint', 'model_box_{}.pth'.format(epoch))),
                           strict=strict
            )
            self.vae_shape.load_state_dict(
                torch.load(os.path.join(exp, 'checkpoint', 'model_shape_{}.pth'.format(epoch))),
                           strict=strict
            )
        elif self.type_ == 'sln':
            self.vae_box.load_state_dict(
                torch.load(os.path.join(exp, 'checkpoint', 'model_box_{}.pth'.format(epoch))),
                strict=False
            )

    def compute_statistics(self, exp, epoch, stats_dataloader, force=False):
        box_stats_f = os.path.join(exp, 'checkpoint', 'model_stats_box_{}.pkl'.format(epoch))
        shape_stats_f = os.path.join(exp, 'checkpoint', 'model_stats_shape_{}.pkl'.format(epoch))

        if self.type_ == 'dis':
            if os.path.exists(box_stats_f) and not force:
                stats = pickle.load(open(box_stats_f, 'rb'))
                self.mean_est_box, self.cov_est_box = stats[0], stats[1]
            else:
                self.mean_est_box, self.cov_est_box = self.vae_box.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est_box, self.cov_est_box], open(box_stats_f, 'wb'))

            if os.path.exists(shape_stats_f) and not force:
                stats = pickle.load(open(shape_stats_f, 'rb'))
                self.mean_est_shape, self.cov_est_shape = stats[0], stats[1]
            else:
                self.mean_est_shape, self.cov_est_shape = self.vae_shape.collect_train_statistics(stats_dataloader,
                                                                                                 with_points=True)
                pickle.dump([self.mean_est_shape, self.cov_est_shape], open(shape_stats_f, 'wb'))

        elif self.type_ == 'shared':
            stats_f = os.path.join(exp, 'checkpoint', 'model_stats_{}.pkl'.format(epoch))
            if os.path.exists(stats_f) and not force:
                stats = pickle.load(open(stats_f, 'rb'))
                self.mean_est, self.cov_est = stats[0], stats[1]
            else:
                self.mean_est, self.cov_est = self.vae.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est, self.cov_est], open(stats_f, 'wb'))

        elif self.type_ == 'sln':
            box_stats_f = os.path.join(exp, 'checkpoint', 'model_stats_{}.pkl'.format(epoch))

            if os.path.exists(box_stats_f) and not force:
                stats = pickle.load(open(box_stats_f, 'rb'))
                self.mean_est_box, self.cov_est_box = stats[0], stats[1]
            else:
                self.mean_est_box, self.cov_est_box = self.vae_box.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est_box, self.cov_est_box], open(box_stats_f, 'wb'))

        elif self.type_ == 'mlp':
            if os.path.exists(box_stats_f) and not force:
                stats = pickle.load(open(box_stats_f, 'rb'))
                self.mean_est_box, self.cov_est_box = stats[0], stats[1]
            else:
                self.mean_est_box, self.cov_est_box = self.vae_box.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est_box, self.cov_est_box], open(box_stats_f, 'wb'))

            if os.path.exists(shape_stats_f) and not force:
                stats = pickle.load(open(shape_stats_f, 'rb'))
                self.mean_est_shape, self.cov_est_shape = stats[0], stats[1]
            else:
                self.mean_est_shape, self.cov_est_shape = self.vae_shape.collect_train_statistics(stats_dataloader)
                pickle.dump([self.mean_est_shape, self.cov_est_shape], open(shape_stats_f, 'wb'))

    def decoder_with_changes_boxes_and_shape(self, z_box, z_shape, objs, triples, attributes, missing_nodes, manipulated_nodes, atlas):
        if self.type_ == 'shared':
            boxes, feats, keep = self.vae.decoder_with_changes(z_box, objs, triples, attributes, missing_nodes, manipulated_nodes)
            points = atlas.forward_inference_from_latent_space(feats, atlas.get_grid())
        elif self.type_ == 'dis' or self.type_ == 'mlp':
            boxes, keep = self.decoder_with_changes_boxes(z_box, objs, triples, attributes, missing_nodes, manipulated_nodes)
            points, _ = self.decoder_with_changes_shape(z_shape, objs, triples, attributes, missing_nodes, manipulated_nodes, atlas)
        elif self.type_ == 'sln':
            (boxes, angles), keep = self.decoder_with_changes_boxes(z_box, objs, triples, attributes, missing_nodes, manipulated_nodes)
            points = None
            boxes = (boxes, angles)

        return boxes, points, keep

    def decoder_with_changes_boxes(self, z, objs, triples, attributes, missing_nodes, manipulated_nodes):
        if self.type_ == 'dis' or self.type_ == 'mlp':
            return self.vae_box.decoder_with_changes(z, objs, triples, attributes, missing_nodes, manipulated_nodes)
        elif self.type_ == 'sln':
            return self.vae_box.decoder_with_additions(z, objs, triples, attributes, missing_nodes, manipulated_nodes,
                                                       (self.mean_est_box, self.cov_est_box))

        if self.type_ == 'shared':
            return None, None

    def decoder_with_changes_shape(self, z, objs, triples, attributes, missing_nodes, manipulated_nodes, atlas):
        if self.type_ == 'dis':
            feats, keep = self.vae_shape.decoder_with_changes(z, objs, triples, attributes, missing_nodes,
                                                                 manipulated_nodes)
        elif self.type_ == 'sln' or self.type_ == 'shared':
            return None, None
        elif self.type_ == 'mlp':
            feats, keep = self.vae_shape.decoder_with_additions(z, objs, missing_nodes, manipulated_nodes)
        return atlas.forward_inference_from_latent_space(feats, atlas.get_grid()), keep

    def decoder_boxes_and_shape(self, z_box, z_shape, objs, triples, attributes, atlas=None):
        angles = None
        if self.type_ == 'shared':
            boxes, angles, feats = self.vae.decoder(z_box, objs, triples, attributes)
            points = atlas.forward_inference_from_latent_space(feats, atlas.get_grid()) if atlas is not None else feats
        elif self.type_ == 'dis' or self.type_ == 'mlp':
            boxes, angles = self.decoder_boxes(z_box, objs, triples, attributes)
            points = self.decoder_shape(z_shape, objs, triples, attributes, atlas)
        elif self.type_ == 'sln':
            boxes, angles = self.decoder_boxes(z_box, objs, triples, attributes)
            points = None

        return boxes, angles, points

    def decoder_boxes(self, z, objs, triples, attributes):
        if self.type_ == 'dis' or self.type_ == 'mlp':
            if self.with_angles:
                return self.vae_box.decoder(z, objs, triples, attributes)
            else:
                return self.vae_box.decoder(z, objs, triples, attributes), None
        elif self.type_ == 'sln':
            return self.vae_box.decoder(z, objs, triples, attributes)
        elif self.type_ == 'shared':
            return None, None

    def decoder_shape(self, z, objs, triples, attributes, atlas=None):
        #print(self.type_)
        if self.type_ == 'dis':
            feats = self.vae_shape.decoder(z, objs, triples, attributes)
        elif self.type_ == 'sln' or self.type_ == 'shared':
            return None, None
        elif self.type_ == 'mlp':
            feats = self.vae_shape.decoder(z, objs)
        return atlas.forward_inference_from_latent_space(feats, atlas.get_grid()) if atlas is not None else feats

    def decoder_with_additions_boxes_and_shape(self, z_box, z_shape, objs, triples, attributes, missing_nodes,
                                               manipulated_nodes, atlas):
        if self.type_ == 'shared':
            outs, keep = self.vae.decoder_with_additions(z_box, objs, triples, attributes, missing_nodes, manipulated_nodes)
            return outs[:2], None, outs[2], keep

        elif self.type_ == 'sln':
            boxes, angles, keep = self.decoder_with_additions_boxs(z_box, objs, triples, attributes, missing_nodes,
                                                                   manipulated_nodes)
            return boxes, angles, None, keep

        elif self.type_ == 'dis' or self.type_ == 'mlp':
            boxes, angles, _ = self.decoder_with_additions_boxs(z_box, objs, triples, attributes, missing_nodes,
                                                                manipulated_nodes)
            points, keep = self.decoder_with_additions_shape(z_shape, objs, triples, attributes, missing_nodes,
                                                             manipulated_nodes, atlas)
        return boxes, angles, points, keep

    def decoder_with_additions_boxs(self, z, objs, triples, attributes, missing_nodes, manipulated_nodes):
        boxes, angles, keep = None, None, None
        if self.type_ == 'dis' or self.type_ == 'mlp':
            boxes, keep = self.vae_box.decoder_with_additions(z, objs, triples, attributes, missing_nodes,
                                                            manipulated_nodes, (self.mean_est_box, self.cov_est_box))
        elif self.type_ == 'sln':
            (boxes, angles), keep = self.vae_box.decoder_with_additions(z, objs, triples, attributes, missing_nodes,
                                                         manipulated_nodes,(self.mean_est_box, self.cov_est_box))
        elif self.type_ == 'shared':
            return  None, None, None
        return boxes, angles, keep

    def decoder_with_additions_shape(self, z, objs, triples, attributes, missing_nodes, manipulated_nodes, atlas):
        if self.type_ == 'sln' or self.type_ == 'shared':
            return None, None
        elif self.type_ == 'dis':
            feats, keep = self.vae_shape.decoder_with_additions(z, objs, triples, attributes, missing_nodes,
                                                                manipulated_nodes)
        elif self.type_ == 'mlp':
            feats, keep = self.vae_shape.decoder_with_additions(z, objs, missing_nodes, manipulated_nodes)

        return atlas.forward_inference_from_latent_space(feats, atlas.get_grid()), keep

    def encode_box_and_shape(self, objs, triples, feats, boxes, angles=None, attributes=None):
        if not self.with_angles:
            angles = None
        if self.type_ == 'dis' or self.type_ == 'mlp':
            return self.encode_box(objs, triples, boxes, angles, attributes), \
                   self.encode_shape(objs, triples, feats, attributes)
        elif self.type_ == 'sln':
            return self.encode_box(objs, triples, boxes, angles, attributes), (None, None)
        elif self.type_ == 'shared':
            with torch.no_grad():
                z, log_var = self.vae.encoder(objs, triples, boxes, feats, attributes, angles)
                return (z, log_var), (z, log_var)

    def encode_shape(self, objs, triples, feats, attributes=None):
        if self.type_ == 'dis':
            z, log_var = self.vae_shape.encoder(objs, triples, feats, attributes)
        elif self.type_ == 'mlp':
            z, log_var = self.vae_shape.encoder(objs, feats)
        elif self.type_ == 'shared':
            return None, None
        elif self.type_ == 'sln':
            return None, None
        return z, log_var

    def encode_box(self, objs, triples, boxes, angles=None, attributes=None):

        if self.type_ == 'dis' or self.type_ == 'mlp':
            z, log_var = self.vae_box.encoder(objs, triples, boxes, attributes, angles)
        elif self.type_ == 'shared':
            return None, None
        elif self.type_ == 'sln':
            z, log_var = self.vae_box.encoder(objs, triples, boxes, angles, attributes)
        return z, log_var

    def sample_box_and_shape(self, point_classes_idx, point_ae, dec_objs, dec_triplets, attributes=None):
        if self.type_ == 'shared':
            return self.vae.sample(point_classes_idx, point_ae, self.mean_est, self.cov_est, dec_objs,  dec_triplets, attributes)
        boxes = self.sample_box(dec_objs, dec_triplets, attributes)
        shapes = self.sample_shape(point_classes_idx, dec_objs, point_ae, dec_triplets, attributes)
        return boxes, shapes

    def sample_box(self, dec_objs, dec_triplets, attributes=None):
        if self.type_ == 'dis' or self.type_ == 'mlp':
            return self.vae_box.sampleBoxes(self.mean_est_box, self.cov_est_box, dec_objs, dec_triplets, attributes)
        if self.type_ == 'sln':
            boxes, angles = self.vae_box.sampleBoxes(self.mean_est_box, self.cov_est_box, dec_objs, dec_triplets, attributes)
            return boxes, angles
        elif self.type_ == 'shared':
            return self.vae.sample(self.mean_est, self.cov_est, dec_objs, dec_triplets, attributes)[0]

    def sample_shape(self, point_classes_idx, dec_objs, point_ae, dec_triplets, attributes=None):
        if self.type_ == 'dis':
            return self.vae_shape.sampleShape(point_classes_idx, point_ae, self.mean_est_shape, self.cov_est_shape,
                                              dec_objs, dec_triplets, attributes)
        if self.type_ == 'mlp':
            return self.vae_shape.sampleShape(point_classes_idx, dec_objs, point_ae, self.mean_est_shape, self.cov_est_shape)
        elif self.type_ == 'shared':
            return self.vae.sample(self.mean_est, self.cov_est, dec_objs, dec_triplets, attributes)[1]
        elif self.type_ == 'sln':
            return None

    def save(self, exp, outf, epoch):
        if self.type_ == 'dis' or self.type_ == 'mlp':
            torch.save(self.vae_box.state_dict(), os.path.join(exp, outf, 'model_box_{}.pth'.format(epoch)))
            torch.save(self.vae_shape.state_dict(), os.path.join(exp, outf, 'model_shape_{}.pth'.format(epoch)))
        elif self.type_ == 'sln':
            torch.save(self.vae_box.state_dict(), os.path.join(exp, outf, 'model_box_{}.pth'.format(epoch)))
        elif self.type_ == 'shared':
            torch.save(self.vae, os.path.join(exp, outf, 'model{}.pth'.format(epoch)))
