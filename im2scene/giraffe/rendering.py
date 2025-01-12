from typing import Any, Union

import torch
import numpy as np
from im2scene.common import interpolate_sphere
from torchvision.utils import save_image, make_grid
import imageio
from math import sqrt, degrees
from os import makedirs
from os.path import join


class Renderer(object):
    '''  Render class for GIRAFFE.

    It provides functions to render the representation.

    Args:
        model (nn.Module): trained GIRAFFE model
        device (device): pytorch device
    '''

    def __init__(self, model, device=None):
        self.model = model.to(device)
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        self.generator = gen

        # sample temperature; only used for visualiations 这个参数决定了batch_size个形状外表的变化程度，如果为0则所有都是一样的形状外表
        self.sample_tmp = 0.65

    def set_random_seed(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def render_full_visualization(self, img_out_path,
                                  render_program=['object_rotation']):
        for rp in render_program:
            if rp == 'object_rotation':
                self.set_random_seed()
                self.render_object_rotation(img_out_path)
            # 自定义一个平移加旋转的变换
            if rp == 'object_wipeout':
                self.set_random_seed()
                self.render_object_wipeout(img_out_path)
            ####
            if rp == 'object_translation_horizontal':
                self.set_random_seed()
                self.render_object_translation_horizontal(img_out_path)
            if rp == 'object_translation_vertical':
                self.set_random_seed()
                self.render_object_translation_depth(img_out_path)
            if rp == 'interpolate_app':
                self.set_random_seed()
                self.render_interpolation(img_out_path)
            if rp == 'interpolate_app_bg':
                self.set_random_seed()
                self.render_interpolation_bg(img_out_path)
            if rp == 'interpolate_shape':
                self.set_random_seed()
                self.render_interpolation(img_out_path, mode='shape')
            if rp == 'object_translation_circle':
                self.set_random_seed()
                self.render_object_translation_circle(img_out_path)
            if rp == 'render_camera_elevation':
                self.set_random_seed()
                self.render_camera_elevation(img_out_path)
            if rp == 'render_camera_rotation':
                self.set_random_seed()
                self.render_camera_rotation(img_out_path)
            if rp == 'render_add_cars':
                self.set_random_seed()
                self.render_add_objects_cars5(img_out_path)
            if rp == 'render_add_clevr10':
                self.set_random_seed()
                self.render_add_objects_clevr10(img_out_path)
            if rp == 'render_add_clevr6':
                self.set_random_seed()
                self.render_add_objects_clevr6(img_out_path)

    def render_object_rotation(self, img_out_path, batch_size=15, n_steps=32):
        # 之前引入的self.generator
        gen = self.generator
        bbox_generator = gen.bounding_box_generator
        # n_boxes为多少？
        n_boxes = bbox_generator.n_boxes

        # Set rotation range 总的旋转程度
        is_full_rotation: Union[bool, Any] = (bbox_generator.rotation_range[0] == 0
                                              and bbox_generator.rotation_range[1] == 1)
        n_steps = int(n_steps * 2) if is_full_rotation else n_steps  # 这里n_steps其实乘以了2，说明是is_full_rotation为True
        r_scale = [0., 1.] if is_full_rotation else [0.1, 0.9]

        # Get Random codes and bg rotation 共有batch_size种车，即shape和appearance
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)

        # Set Camera
        camera_matrices = gen.get_camera(batch_size=batch_size)
        # 缩放部分 无缩放
        s_val = [[0, 0, 0] for i in range(n_boxes)]
        # 平移部分 一个固定的平移
        t_val = [[0.5, 0.5, 0.5] for i in range(n_boxes)]
        # 旋转部分 n_boxes相当于有几个Object
        r_val = [0. for i in range(n_boxes)]

        # 见giraffe/im2scene/model/giraffe/models/generators，因为这部分是纯旋转，故缩放与平移以及固定下来，最后在加上旋转
        s, t, _ = gen.get_transformations(s_val, t_val, r_val, batch_size)

        out = []

        # test 记录生成次数
        i = 0

        # 记录每一次与输入图片的最好loss
        best_loss = 0
        # step其实说明整个旋转矩阵使用了n_steps个，每一次绕z轴的旋转矩阵都会发生一点变化，具体图片中产生几种旋转的小图由n_image参数决定
        for step in range(n_steps):
            # Get rotation for this step 注意由于我们的n_boxes为1，故这里r其实相当于一个只有一个元素的列表
            # 注意一开始step为0
            r = [step * 1.0 / (n_steps - 1) for i in range(n_boxes)]
            r = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r]
            # 计算角度
            angle = r[0] * 2 * np.pi
            print(degrees(angle))
            # print(r)  # 此时生成的r为一个只包含一个元素的列表[]
            # 获得旋转矩阵 最终计算旋转矩阵其实在camera.py中
            r = gen.get_rotation(r, batch_size)
            # test部分
            # print(r)
            # print(r.shape)    # torch.Size([15, 1, 3, 3]) 这里的15是来自于batch_size=15，一次性喂入15个旋转矩阵，对应之后生成的15张大图片,15的R其实都一样
            # print(r[1].reshape(3, 3))
            # print((r[1].reshape(3, 3)).shape)  # 生成torch.Size([3, 3])旋转矩阵
            i = i + 1

            # define full transformation and evaluate model 每一次生成的不同旋转加上之前固定的平移和缩放构成了每一次的变换，共2*n_steps次
            transformations = [s, t, r]
            with torch.no_grad():
                # 将变换传递给Generator来做体渲染和神经渲染, latent_codes是物体和背景的外观和形状编码
                # 每一次生成一组相同的旋转矩阵，结合缩放平移生成变换，然后喂入生成器产生一个新的out_i
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='val')
                ######
                # print(out_i)  # 输出测试
                # print(out_i.shape)  # 每一次的图片 torch.Size([15, 3, 256, 256])，15是batch，3为channel
                ######

            out.append(out_i.cpu())

        print(i)  # i为64

        out = torch.stack(out)
        # 建立输出目录out
        out_folder = join(img_out_path, 'rotation_object')
        makedirs(out_folder, exist_ok=True)
        # 保存图片视频  def save_video_and_images(self, imgs, out_folder....
        # 最后生成的图片系列有15张，每一张里面包含了六种旋转图像的小图像
        # 各个角度的都算完了，之后再保存
        self.save_video_and_images(
            out, out_folder, name='rotation_object',
            is_full_rotation=is_full_rotation,
            add_reverse=(not is_full_rotation))

    def render_object_translation_horizontal(self, img_out_path, batch_size=15,
                                             n_steps=32):
        gen = self.generator

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)
        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]
             for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            x_val = 0.5
        elif n_boxes == 2:
            t = [[0.5, 0.5, 0.]]
            x_val = 1.

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[x_val, i, 0.]]
            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        out_folder = join(img_out_path, 'translation_object_horizontal')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='translation_horizontal',
            add_reverse=True)

    # 自定义的旋转加平移变换
    def render_object_wipeout(self, img_out_path, batch_size=15, n_steps=512):
        gen = self.generator

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)
        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]  # 无缩放
             for i in range(n_boxes)]
        n_steps = int(n_steps * 2)
        # 最小为0最大为1
        r_scale = [0., 1.]

        if n_boxes == 1:
            t = []
            x_val = 0.5
        elif n_boxes == 2:
            t = [[0.5, 0.5, 0.]]
            x_val = 1.0

        out = []
        for step in range(n_steps):
            # translation 变化的平移，与纯旋转可以对比
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[0.1, i, 0.]]
            # rotation 跟纯旋转一样
            r = [step * 1.0 / (n_steps - 1) for i in range(n_boxes)]
            r = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r]

            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        out_folder = join(img_out_path, 'object_wipeout')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='object_wipeout',
            add_reverse=True)
        ###

    def render_object_translation_depth(self, img_out_path, batch_size=15,
                                        n_steps=32):
        gen = self.generator
        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)

        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]
             for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            y_val = 0.5
        elif n_boxes == 2:
            t = [[0.4, 0.8, 0.]]
            y_val = 0.2

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[i, y_val, 0.]]
            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)
        out_folder = join(img_out_path, 'translation_object_depth')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='translation_depth', add_reverse=True)

    def render_interpolation(self, img_out_path, batch_size=15, n_samples=6,
                             n_steps=32, mode='app'):
        gen = self.generator
        n_boxes = gen.bounding_box_generator.n_boxes

        # Get values
        z_shape_obj_1, z_app_obj_1, z_shape_bg_1, z_app_bg_1 = \
            gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        z_i = [
            gen.sample_z(
                z_app_obj_1.shape,
                tmp=self.sample_tmp) for j in range(n_samples)
        ]

        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)

        if n_boxes == 1:
            t_val = [[0.5, 0.5, 0.5]]
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            t_val,
            [0.5 for i in range(n_boxes)],
            batch_size
        )

        out = []
        for j in range(n_samples):
            z_i1 = z_i[j]
            z_i2 = z_i[(j + 1) % (n_samples)]
            for step in range(n_steps):
                w = step * 1.0 / ((n_steps) - 1)
                z_ii = interpolate_sphere(z_i1, z_i2, w)
                if mode == 'app':
                    latent_codes = [z_shape_obj_1, z_ii, z_shape_bg_1,
                                    z_app_bg_1]
                else:
                    latent_codes = [z_ii, z_app_obj_1, z_shape_bg_1,
                                    z_app_bg_1]
                with torch.no_grad():
                    out_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, bg_rotation, mode='val')
                out.append(out_i.cpu())
        out = torch.stack(out)

        # Save Video
        out_folder = join(img_out_path, 'interpolate_%s' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate_%s' % mode,
            is_full_rotation=True)

    def render_interpolation_bg(self, img_out_path, batch_size=15, n_samples=6,
                                n_steps=32, mode='app'):
        gen = self.generator
        n_boxes = gen.bounding_box_generator.n_boxes

        # Get values
        z_shape_obj_1, z_app_obj_1, z_shape_bg_1, z_app_bg_1 = \
            gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        z_i = [
            gen.sample_z(
                z_app_bg_1.shape,
                tmp=self.sample_tmp) for j in range(n_samples)
        ]

        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)

        if n_boxes == 1:
            t_val = [[0.5, 0.5, 0.5]]
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            t_val,
            [0.5 for i in range(n_boxes)],
            batch_size
        )

        out = []
        for j in range(n_samples):
            z_i1 = z_i[j]
            z_i2 = z_i[(j + 1) % (n_samples)]
            for step in range(n_steps):
                w = step * 1.0 / ((n_steps) - 1)
                z_ii = interpolate_sphere(z_i1, z_i2, w)
                if mode == 'app':
                    latent_codes = [z_shape_obj_1, z_app_obj_1, z_shape_bg_1,
                                    z_ii]
                else:
                    latent_codes = [z_shape_obj_1, z_app_obj_1, z_ii,
                                    z_app_bg_1]
                with torch.no_grad():
                    out_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, bg_rotation, mode='val')
                out.append(out_i.cpu())
        out = torch.stack(out)

        # Save Video
        out_folder = join(img_out_path, 'interpolate_bg_%s' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate_bg_%s' % mode,
            is_full_rotation=True)

    def render_object_translation_circle(self, img_out_path, batch_size=15,
                                         n_steps=32):
        gen = self.generator

        # Disable object sampling
        sample_object_existance = gen.sample_object_existance
        gen.sample_object_existance = False

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)
        n_boxes = gen.bounding_box_generator.n_boxes

        s = [[0, 0, 0, ]
             for i in range(n_boxes)]
        r = [0 for i in range(n_boxes)]
        s10, t10, r10 = gen.get_random_transformations(batch_size)

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            cos_i = (np.cos(2 * np.pi * i) * 0.5 + 0.5).astype(np.float32)
            sin_i = (np.sin(2 * np.pi * i) * 0.5 + 0.5).astype(np.float32)
            if n_boxes <= 2:
                t = [[0.5, 0.5, 0.] for i in range(n_boxes - 1)] + [
                    [cos_i, sin_i, 0]
                ]
                transformations = gen.get_transformations(s, t, r, batch_size)
            else:
                cos_i, sin_i = cos_i * 1.0 - 0.0, sin_i * 1. - 0.
                _, ti, _ = gen.get_transformations(
                    val_t=[[cos_i, sin_i, 0]], batch_size=batch_size)
                t10[:, -1:] = ti
                transformations = [s10, t10, r10]

            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        gen.sample_object_existance = sample_object_existance

        # Save Video
        out_folder = join(img_out_path, 'translation_circle')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(out, out_folder, name='translation_circle',
                                   is_full_rotation=True)

    def render_camera_elevation(self, img_out_path, batch_size=15, n_steps=64):
        # 只有相机高程的调整
        # 跟旋转一样的代码
        gen = self.generator
        n_boxes = gen.bounding_box_generator.n_boxes
        #
        r_range = [0.1, 0.9]

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        transformations = gen.get_transformations(
            # 在这种情况下实际上transformations(即对物体使用固定的单一仿射变换)
            # 物体缩放部分  s_val
            [[0., 0., 0.] for i in range(n_boxes)],
            # 物体平移部分 t_val
            [[0.5, 0.5, 0.5] for i in range(n_boxes)],
            # 物体旋转部分 r_val
            [0.5 for i in range(n_boxes)],
            batch_size,
        )

        out = []
        for step in range(n_steps):
            # 将0-1划分为(n_steps - 1)段
            v = step * 1.0 / (n_steps - 1)
            # r = 0.1 + v * (0.9 - 0.1)
            r = r_range[0] + v * (r_range[1] - r_range[0])
            # 注意这里向相机多传入了一个参数r
            camera_matrices = gen.get_camera(val_v=r, batch_size=batch_size)
            # 以下都是一样的
            with torch.no_grad():
                out_i = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        out_folder = join(img_out_path, 'camera_elevation')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(out, out_folder, name='elevation_camera',
                                   is_full_rotation=False)

    def render_camera_rotation(self, img_out_path, batch_size=15, n_steps=64):
        # 只有相机高程的调整
        # 跟旋转一样的代码
        gen = self.generator
        n_boxes = gen.bounding_box_generator.n_boxes
        #
        r_range = [0.1, 0.9]

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        # 这里的sample_tmp值在最前面指定，可以控制在不同的种类里，形状外表的变化幅度，为0则表示所有都是一样的形状外表
        # bg_rotation = gen.get_random_bg_rotation(batch_size)
        bg_rotation = gen.get_bg_rotation(val=0, batch_size=batch_size)
        # 固定背景旋转，但其实在这里get_random_bg_rotation(batch_size)也是一样的效果，因为get_random_bg_rotation函数默认的范围就是[0,0]，即背景无旋转
        transformations = gen.get_transformations(
            # 在这种情况下实际上transformations(即对物体使用固定的单一仿射变换)
            # 物体缩放部分  s_val
            [[0., 0., 0.] for i in range(n_boxes)],
            # 物体平移部分 t_val
            [[0.5, 0.5, 0.5] for i in range(n_boxes)],
            # [[0., 0., 0.] for i in range(n_boxes)],
            # 物体旋转部分 r_val
            [0. for i in range(n_boxes)],
            # [0. for i in range(n_boxes)],
            batch_size,
        )

        out = []
        for step in range(n_steps):
            # 将0-1划分为(n_steps - 1)段
            # v = step * 1.0 / (n_steps - 1)
            # 自己加的部分 每一个step都会取一个递增的u
            u = step * 1.0 / (n_steps - 1)
            #
            # r = 0.1 + v * (0.9 - 0.1) 这里的r与generator中的没有关系，不要搞混,仅仅只是一个程度系数，每一个step会得到一个递增的r值
            r = r_range[0] + u * (r_range[1] - r_range[0])
            # 注意这里向相机多传入了一个参数r
            # 参数为val_u为控制相机旋转；参数为val_v为控制相机高程；参数为val_r为radial？
            camera_matrices = gen.get_camera(val_u=r, batch_size=batch_size)
            # 注意get_camera返回的其实是一个二维元组，包含了get_camera生成的camera_mat, world_mat两个矩阵
            # print(len(camera_matrices))    输出2
            # 以下都是一样的
            with torch.no_grad():
                out_i = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val', it=0,
                    return_alpha_map=False,
                    not_render_background=False,
                    only_render_background=False)
                # 这里是generator中的def forward！Generator类是一个nn类，会自动调用forward，原因不言自明
            out.append(out_i.cpu())
        out = torch.stack(out)

        out_folder = join(img_out_path, 'camera_rotation')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(out, out_folder, name='camera_rotation',
                                   is_full_rotation=False)

    def render_add_objects_cars5(self, img_out_path, batch_size=15):
        # 增加Object（车）数目
        gen = self.generator

        # Get values
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = gen.get_latent_codes(
            batch_size, tmp=self.sample_tmp)
        z_shape_obj = gen.sample_z(
            z_shape_obj[:, :1].repeat(1, 6, 1).shape, tmp=self.sample_tmp)
        z_app_obj = gen.sample_z(
            z_app_obj[:, :1].repeat(1, 6, 1).shape, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(val_v=0., batch_size=batch_size)

        s = [
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
        ]

        t = [
            [-0.7, -.8, 0.],
            [-0.7, 0.5, 0.],
            [-0.7, 1.8, 0.],
            [1.5, -.8, 0.],
            [1.5, 0.5, 0.],
            [1.5, 1.8, 0.],
        ]
        r = [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
        outs = []
        for i in range(1, 7):
            transformations = gen.get_transformations(
                s[:i], t[:i], r[:i], batch_size)
            latent_codes = [z_shape_obj[:, :i], z_app_obj[:, :i], z_shape_bg,
                            z_app_bg]
            with torch.no_grad():
                out = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val').cpu()
            outs.append(out)
        outs = torch.stack(outs)
        idx = torch.arange(6).reshape(-1, 1).repeat(1, (128 // 6)).reshape(-1)
        outs = outs[[idx]]

        # import pdb; pdb.set_trace()
        out_folder = join(img_out_path, 'add_cars')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(outs, out_folder, name='add_cars',
                                   is_full_rotation=False, add_reverse=True)

    def render_add_objects_clevr10(self, img_out_path, batch_size=15):
        gen = self.generator

        # Disable object sampling
        sample_object_existance = gen.sample_object_existance
        gen.sample_object_existance = False

        n_steps = 6
        n_objs = 12

        # Get values
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = gen.get_latent_codes(
            batch_size, tmp=self.sample_tmp)
        z_shape_obj = gen.sample_z(
            z_shape_obj[:, :1].repeat(1, n_objs, 1).shape, tmp=self.sample_tmp)
        z_app_obj = gen.sample_z(
            z_app_obj[:, :1].repeat(1, n_objs, 1).shape, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(val_v=0., batch_size=batch_size)

        s = [
            [0, 0, 0] for i in range(n_objs)
        ]
        t = []
        for i in range(n_steps):
            if i % 3 == 0:
                x = 0.0
            elif i % 3 == 1:
                x = 0.5
            else:
                x = 1

            if i in [0, 1, 2]:
                y = 0.
            else:
                y = 0.8
            t = t + [[x, y, 0], [x, y + 0.4, 0]]
        r = [
            0 for i in range(n_objs)
        ]
        out_total = []
        for i in range(2, n_objs + 1, 2):
            transformations = gen.get_transformations(
                s[:i], t[:i], r[:i], batch_size)
            latent_codes = [z_shape_obj[:, :i], z_app_obj[:, :i], z_shape_bg,
                            z_app_bg]
            with torch.no_grad():
                out = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val').cpu()
            out_total.append(out)
        out_total = torch.stack(out_total)
        idx = torch.arange(6).reshape(-1, 1).repeat(1, (128 // 6)).reshape(-1)
        outs = out_total[[idx]]

        gen.sample_object_existance = sample_object_existance

        out_folder = join(img_out_path, 'add_clevr_objects10')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(outs, out_folder, name='add_clevr10',
                                   is_full_rotation=False, add_reverse=True)

    def render_add_objects_clevr6(self, img_out_path, batch_size=15):

        gen = self.generator

        # Disable object sampling
        sample_object_existance = gen.sample_object_existance
        gen.sample_object_existance = False

        n_objs = 6
        # Get values
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = gen.get_latent_codes(
            batch_size, tmp=self.sample_tmp)
        z_shape_obj = gen.sample_z(
            z_shape_obj[:, :1].repeat(1, n_objs, 1).shape, tmp=self.sample_tmp)
        z_app_obj = gen.sample_z(
            z_app_obj[:, :1].repeat(1, n_objs, 1).shape, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(val_v=0., batch_size=batch_size)

        s = [
            [0, 0, 0] for i in range(n_objs)
        ]
        t = []
        for i in range(n_objs):
            if i % 2 == 0:
                x = 0.2
            else:
                x = 0.8

            if i in [0, 1]:
                y = 0.
            elif i in [2, 3]:
                y = 0.5
            else:
                y = 1.
            t = t + [[x, y, 0]]
        r = [
            0 for i in range(n_objs)
        ]
        out_total = []
        for i in range(1, n_objs + 1):
            transformations = gen.get_transformations(
                s[:i], t[:i], r[:i], batch_size)
            latent_codes = [z_shape_obj[:, :i], z_app_obj[:, :i], z_shape_bg,
                            z_app_bg]
            with torch.no_grad():
                out = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val').cpu()
                out_total.append(out)
        out_total = torch.stack(out_total)
        idx = torch.arange(6).reshape(-1, 1).repeat(1, (128 // 6)).reshape(-1)
        outs = out_total[[idx]]

        gen.sample_object_existance = sample_object_existance

        out_folder = join(img_out_path, 'add_clevr_objects6')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(outs, out_folder, name='add_clevr6',
                                   is_full_rotation=False, add_reverse=True)

    ##################
    # Helper functions
    def write_video(self, out_file, img_list, n_row=5, add_reverse=False,
                    write_small_vis=True):
        n_steps, batch_size = img_list.shape[:2]
        nrow = n_row if (n_row is not None) else int(sqrt(batch_size))
        img = [(255 * make_grid(img, nrow=nrow, pad_value=1.).permute(
            1, 2, 0)).cpu().numpy().astype(np.uint8) for img in img_list]
        if add_reverse:
            img += list(reversed(img))
        imageio.mimwrite(out_file, img, fps=30, quality=8)
        if write_small_vis:
            img = [(255 * make_grid(img, nrow=batch_size, pad_value=1.).permute(
                1, 2, 0)).cpu().numpy().astype(
                np.uint8) for img in img_list[:, :9]]
            if add_reverse:
                img += list(reversed(img))
            imageio.mimwrite(
                (out_file[:-4] + '_sm.mp4'), img, fps=30, quality=4)

    def save_video_and_images(self, imgs, out_folder, name='rotation_object',
                              is_full_rotation=False, img_n_steps=6,
                              add_reverse=False):
        # img_n_steps=6 这个参数决定了一张图片里面有多少个旋转的小图片
        out_file_video = join(out_folder, '%s.mp4' % name)

        # Save video 见上
        self.write_video(out_file_video, imgs, add_reverse=add_reverse)

        # Save images
        n_steps, batch_size = imgs.shape[:2]
        if is_full_rotation:
            idx_paper = np.linspace(
                0, n_steps - n_steps // img_n_steps, img_n_steps
            ).astype(np.int)
            # 本质上即从n_steps中平均抽取了img_n_steps张
        else:
            idx_paper = np.linspace(0, n_steps - 1, img_n_steps).astype(np.int)
        for idx in range(batch_size):
            # batch_size 即前面的shape和a篇appearance车型说明
            img_grid = imgs[idx_paper, idx]
            save_image(make_grid(
                img_grid, nrow=img_n_steps, pad_value=1.), join(
                out_folder, '%04d_%s.jpg' % (idx, name)))
