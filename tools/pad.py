from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
import torch


def pad_verts_faces_simple(self, batch):
    verts = [v for v, f, vdf, l ,f_infos, m, s in batch]
    faces = [f for v, f, vdf, l ,f_infos, m, s  in batch]
    verts_data_faces = [vdf for v, f, vdf, l ,f_infos, m, s in batch]
    labels = [l for v, f, vdf, l ,f_infos, m, s in batch]
    f_infos = [f_infos for v, f, vdf, l ,f_infos, m, s in batch]
    mean = [m for v, f, vdf, l ,f_infos, m, s in batch]
    scale = [s for v, f, vdf, l ,f_infos, m, s in batch]

    verts = pad_sequence(verts, batch_first=True, padding_value=0)
    faces = pad_sequence(faces, batch_first=True, padding_value=-1)
    verts_data_faces = torch.cat(verts_data_faces)
    labels = torch.cat(labels)
    f_infos = torch.cat(f_infos)
    mean = torch.cat(mean)
    scale = torch.cat(scale)

    return verts, faces, verts_data_faces, labels, f_infos, mean, scale

def pad_verts_faces(self, batch):
    labeled_fibers = ()
    tractography_fibers = ()
    for i in range(len(batch)):
        labeled_fibers += (batch[i][0],)
        tractography_fibers += (batch[i][1],)

    verts_lf = [v for v, f, vdf, l, f_infos, m, s in labeled_fibers]
    faces_lf = [f for v, f, vdf, l, f_infos, m, s in labeled_fibers]
    verts_data_faces_lf = [vdf for v, f, vdf, l, f_infos, m, s in labeled_fibers]
    labels_lf = [l for v, f, vdf, l, f_infos, m, s in labeled_fibers]
    f_infos_lf = [f_infos for v, f, vdf, l, f_infos, m, s in labeled_fibers]
    mean_lf = [m for v, f, vdf, l, f_infos, m, s in labeled_fibers]
    scale_lf = [s for v, f, vdf, l, f_infos, m, s in labeled_fibers]

    verts_tf = [v for v, f, vdf, l, f_infos, m, s in tractography_fibers]
    faces_tf = [f for v, f, vdf, l, f_infos, m, s in tractography_fibers]
    verts_data_faces_tf = [vdf for v, f, vdf, l, f_infos, m, s in tractography_fibers]
    labels_tf = [l for v, f, vdf, l, f_infos, m, s in tractography_fibers]
    f_infos_tf = [f_infos for v, f, vdf, l, f_infos, m, s in tractography_fibers]
    mean_tf = [m for v, f, vdf, l, f_infos, m, s in tractography_fibers]
    scale_tf = [s for v, f, vdf, l, f_infos, m, s in tractography_fibers]

    verts = verts_lf + verts_tf
    faces = faces_lf + faces_tf
    verts_data_faces = verts_data_faces_lf + verts_data_faces_tf
    labels = labels_lf + labels_tf
    f_infos = f_infos_lf + f_infos_tf
    mean = mean_lf + mean_tf
    scale = scale_lf + scale_tf
    verts = pad_sequence(verts, batch_first=True, padding_value=0.0)
    faces = pad_sequence(faces, batch_first=True, padding_value=-1)

    verts_data_faces = torch.cat(verts_data_faces)
    labels = torch.cat(labels)
    f_infos = torch.cat(f_infos)
    mean = torch.cat(mean)
    scale = torch.cat(scale)

    return verts, faces, verts_data_faces, labels, f_infos, mean, scale