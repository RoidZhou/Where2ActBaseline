from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import examples


# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap-40147-model_epoch_269-nothing/pred.feats'
kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap-102860-model_epoch_269-nothing/pred.feats'
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap-model_epoch_269-nothing/pred.feats'
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap-model_epoch_269-nothing/pred.pts'
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap-model_epoch_269-nothing/pred.feats'
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_grasp/visu_critic_heatmap_custom-40147-model_epoch_76-nothing/pred.feats'
# kitti_file = '/home/zhou/autolab/Where2ActBaseline/code/logs/model_3d_critic_pushing/visu_critic_heatmap_custom-model_epoch_269-nothing/pred.feats'
# pointcloud = np.fromfile(file=kitti_file, dtype=np.float32, count=-1).reshape([-1, 3])
# pointcloud = np.loadtxt(kitti_file).reshape([-1, 3])
pointcloud = np.loadtxt(kitti_file).reshape([-1, 4])
pc_mask = pointcloud[:, 3]>0
pc_mask_matrix = np.repeat(pc_mask, 3, axis=0)
pc_mask_matrix = pc_mask_matrix.reshape(-1, 3)
pred_pc = pointcloud[:, 0:3]
pred_pc_new = pointcloud[:, :3][pc_mask_matrix]
pred_pc_new = pred_pc_new.reshape(-1, 3)

# Basic Plot
pv.plot(pred_pc)
pv.plot(pointcloud[:, :3])
pv.global_theme.allow_empty_mesh = True
# Plot with Scalars
pv.plot(
    pred_pc_new[:, :3],
    scalars=pred_pc_new[:, :3][:, 2],
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False,
)

pred_pc_new = pv.PolyData(pred_pc_new)
pred_pc = pv.PolyData(pred_pc)


pl = pv.Plotter()
pl.add_mesh(pred_pc, color='#0000FF', show_edges=True)
pl.add_mesh(pred_pc_new, color='#FF0000', show_edges=True)
pl.show()