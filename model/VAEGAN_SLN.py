import torch
import torch.nn as nn
import torch.nn.functional as F

from model.graph import GraphTripleConvNet, _init_weights, make_mlp

import numpy as np


class Sg2ScVAEModel(nn.Module):
    def __init__(self, vocab, embedding_dim=128, batch_size=32,
                 train_3d=True,
                 decoder_cat=False,
                 Nangle=24,
                 gconv_mode='feedforward',
                 gconv_pooling='avg', gconv_num_layers=5,
                 mlp_normalization='none',
                 vec_noise_dim=0,
                 layout_noise_dim=0,
                 use_AE=False,
                 use_attr=True,
                 with_angle=True,
                 residual=False,
                 num_attrs=0):

        super(Sg2ScVAEModel, self).__init__()

        self.with_angle = with_angle

        gconv_dim = embedding_dim
        gconv_hidden_dim = gconv_dim * 4

        if self.with_angle:
            box_embedding_dim = int(embedding_dim * 3 / 4)
            angle_embedding_dim = int(embedding_dim / 4)
        else:
            box_embedding_dim = int(embedding_dim)

        attr_embedding_dim = 0
        obj_embedding_dim = embedding_dim

        self.use_attr = use_attr
        self.batch_size = batch_size
        self.train_3d = train_3d
        self.decoder_cat = decoder_cat
        self.vocab = vocab
        self.vec_noise_dim = vec_noise_dim
        #self.layout_noise_dim = layout_noise_dim
        self.use_AE = use_AE

        if self.use_attr:
            obj_embedding_dim = int(embedding_dim * 3 / 4)
            attr_embedding_dim = int(embedding_dim / 4)

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])
        #num_attrs = len(vocab['attrib_idx_to_name'])

        # making nets
        self.obj_embeddings_ec = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_ec = nn.Embedding(num_preds, embedding_dim * 2)
        self.obj_embeddings_dc = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim)
        if use_attr:
            self.attr_embedding_ec = nn.Embedding(num_attrs, attr_embedding_dim)
            self.attr_embedding_dc = nn.Embedding(num_attrs, attr_embedding_dim)
        if self.decoder_cat:
            self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim * 2)
        if self.train_3d:
            self.box_embeddings = nn.Linear(6, box_embedding_dim)
        else:
            self.box_embeddings = nn.Linear(4, box_embedding_dim)
        if self.with_angle:
            self.angle_embeddings = nn.Embedding(Nangle, angle_embedding_dim)
        # weight sharing of mean and var
        self.box_mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                     batch_norm=mlp_normalization)
        self.box_mean = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.box_var = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        if self.with_angle:
            self.angle_mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                       batch_norm=mlp_normalization)
            self.angle_mean = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)
            self.angle_var = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)        # graph conv net
        self.gconv_net_ec = None
        self.gconv_net_dc = None

        gconv_kwargs_ec = {
            'input_dim_obj': gconv_dim * 2,
            'input_dim_pred': gconv_dim * 2,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        gconv_kwargs_dc = {
            'input_dim_obj': gconv_dim,
            'input_dim_pred': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers,
            'mlp_normalization': mlp_normalization,
            'residual': residual
        }
        if self.decoder_cat:
            gconv_kwargs_dc['input_dim_obj'] = gconv_dim * 2
            gconv_kwargs_dc['input_dim_pred'] = gconv_dim * 2

        self.gconv_net_ec = GraphTripleConvNet(**gconv_kwargs_ec)
        self.gconv_net_dc = GraphTripleConvNet(**gconv_kwargs_dc)

        # box prediction net
        if self.train_3d:
            box_net_dim = 6
        else:
            box_net_dim = 4
        box_net_layers = [gconv_dim * 2, gconv_hidden_dim, box_net_dim]
        if self.use_attr:
            box_net_layers = [gconv_dim * 2 + attr_embedding_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = make_mlp(box_net_layers, batch_norm=mlp_normalization, norelu=True)

        if self.with_angle:
            # angle prediction net
            angle_net_layers = [gconv_dim * 2, gconv_hidden_dim, Nangle]
            self.angle_net = make_mlp(angle_net_layers, batch_norm=mlp_normalization, norelu=True)

        # initialization
        self.box_embeddings.apply(_init_weights)
        self.box_mean_var.apply(_init_weights)
        self.box_mean.apply(_init_weights)
        self.box_var.apply(_init_weights)
        if self.with_angle:
            self.angle_mean_var.apply(_init_weights)
            self.angle_mean.apply(_init_weights)
            self.angle_var.apply(_init_weights)
        self.box_net.apply(_init_weights)

        self.p_box = 0.25

    def encoder(self, objs, triples, boxes_gt, angles_gt, attributes, box_keep=None):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        boxes_in = boxes_gt
        angles_in = angles_gt

        obj_vecs = self.obj_embeddings_ec(objs)
        if self.use_attr:
            attr_vecs = self.attr_embedding_ec(attributes)
            obj_vecs = torch.cat([obj_vecs, attr_vecs], dim=1)

        pred_vecs = self.pred_embeddings_ec(p)
        boxes_vecs = self.box_embeddings(boxes_in)

        if self.with_angle:
            angle_vecs = self.angle_embeddings(angles_in)
            obj_vecs = torch.cat([obj_vecs, boxes_vecs, angle_vecs], dim=1)
        else:
            obj_vecs = torch.cat([obj_vecs, boxes_vecs], dim=1)

        if self.gconv_net_ec is not None:
            obj_vecs, pred_vecs = self.gconv_net_ec(obj_vecs, pred_vecs, edges)

        obj_vecs_box = self.box_mean_var(obj_vecs)
        mu_box = self.box_mean(obj_vecs_box)
        logvar_box = self.box_var(obj_vecs_box)

        if self.with_angle:
            obj_vecs_angle = self.angle_mean_var(obj_vecs)
            mu_angle = self.angle_mean(obj_vecs_angle)
            logvar_angle = self.angle_var(obj_vecs_angle)
            mu = torch.cat([mu_box, mu_angle], dim=1)
            logvar = torch.cat([logvar_box, logvar_angle], dim=1)
        else:
            mu = mu_box
            logvar = logvar_box
        outputs = [mu, logvar]

        return outputs

    def decoder_with_additions(self, z, objs, triples, attributes, missing_nodes, manipulated_nodes, distribution=None):
        nodes_added = []
        if distribution is not None:
            mu, cov = distribution

        for i in range(len(missing_nodes)):
          ad_id = missing_nodes[i] + i
          nodes_added.append(ad_id)
          noise = np.zeros(z.shape[1])
          if distribution is not None:
              zeros = torch.from_numpy(np.random.multivariate_normal(mu, cov, 1)).float().cuda()
          else:
              zeros = torch.from_numpy(noise.reshape(1, z.shape[1]))
          zeros.requires_grad = True
          zeros = zeros.float().cuda()
          z = torch.cat([z[:ad_id], zeros, z[ad_id:]], dim=0)

        keep = []
        for i in range(len(z)):
            if i not in nodes_added and i not in manipulated_nodes:
                keep.append(1)
            else:
                keep.append(0)

        keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

        return self.decoder(z, objs, triples, attributes), keep

    def decoder(self, z, objs, triples, attributes):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_dc(objs)
        if self.use_attr:
            attr_vecs = self.attr_embedding_dc(attributes)
            obj_vecs = torch.cat([obj_vecs, attr_vecs], dim=1)
        pred_vecs = self.pred_embeddings_dc(p)

        # concatenate noise first
        if self.decoder_cat:
            obj_vecs = torch.cat([obj_vecs, z], dim=1)
            obj_vecs, pred_vecs = self.gconv_net_dc(obj_vecs, pred_vecs, edges)

        # concatenate noise after gconv
        else:
            obj_vecs, pred_vecs = self.gconv_net_dc(obj_vecs, pred_vecs, edges)
            obj_vecs = torch.cat([obj_vecs, z], dim=1)

        if self.use_attr:
            obj_vecs_box = torch.cat([obj_vecs, attr_vecs], dim=1)
            boxes_pred = self.box_net(obj_vecs_box)
        else:
            boxes_pred = self.box_net(obj_vecs)
        if self.with_angle:
            angles_pred = F.log_softmax(self.angle_net(obj_vecs), dim=1)
            return boxes_pred, angles_pred
        else:
            return boxes_pred

    def forward(self, objs, triples, boxes_gt, angles_gt, attributes, obj_to_img, box_keep=None):
        encoder_outputs = self.encoder(objs, triples, boxes_gt, angles_gt, attributes, box_keep=box_keep)

        mu, logvar = encoder_outputs[:2]

        if self.use_AE:
            z = mu
        else:
            # reparameterization
            std = torch.exp(0.5*logvar)
            # standard sampling
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)

            # You can theoretically also just sample one per room, instead of one per obj
            # Just replicate it
        if self.with_angle:
            boxes_pred, angles_pred = self.decoder(z, objs, triples, attributes)
            outputs =  [mu, logvar, boxes_pred, angles_pred]
        else:
            boxes_pred = self.decoder(z, objs, triples, attributes)
            outputs = [mu, logvar, boxes_pred]

        return outputs

    def sampleBoxes(self, mean_est, cov_est, dec_objs, dec_triplets, attributes=None):
        with torch.no_grad():
            z = torch.from_numpy(
            np.random.multivariate_normal(mean_est, cov_est, dec_objs.size(0))).float().cuda()

            return self.decoder(z, dec_objs, dec_triplets, attributes)

    def collect_train_statistics(self, train_loader):
        mean_cat = None
        for idx, data in enumerate(train_loader):
            if data == -1:
                continue
            try:
                objs, triples, tight_boxes, objs_to_scene, triples_to_scene = data['decoder']['objs'], \
                                                                              data['decoder']['tripltes'], \
                                                                              data['decoder']['boxes'], \
                                                                              data['decoder']['obj_to_scene'], \
                                                                              data['decoder']['tiple_to_scene']
            except Exception as e:
                print('Exception', str(e))
                continue

            objs, triples, tight_boxes = objs.cuda(), triples.cuda(), tight_boxes.cuda()
            boxes = tight_boxes[:, :6]
            attributes = None

            angles = tight_boxes[:, 6].long() - 1
            angles = torch.where(angles > 0, angles, torch.zeros_like(angles))

            mean, logvar = self.encoder(objs, triples, boxes, angles, attributes)

            mean = mean.data.cpu().clone()
            if mean_cat is None:
                mean_cat = mean.numpy()
            else:
                mean_cat = np.concatenate([mean_cat, mean.numpy()], axis=0)

        mean_est = np.mean(mean_cat, axis=0, keepdims=True)  # size 1*embed_dim
        mean_cat = mean_cat - mean_est
        n = mean_cat.shape[0]
        d = mean_cat.shape[1]
        cov_est = np.zeros((d, d))
        for i in range(n):
            x = mean_cat[i]
            cov_est += 1.0 / (n - 1.0) * np.outer(x, x)
        mean_est = mean_est[0]

        return mean_est, cov_est
