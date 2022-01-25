import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot


def get_camera_mat(fov=49.13, invert=True):
    # 相机内参矩阵
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
    # in our case, sensor = 2 as pixels are in [-1, 1]
    focal = 1. / np.tan(0.5 * fov * np.pi/180.)
    focal = focal.astype(np.float32)
    mat = torch.tensor([
        [focal, 0., 0., 0.],
        [0., focal, 0., 0.],
        [0., 0., 1, 0.],
        [0., 0., 0., 1.]
    ]).reshape(1, 4, 4)

    if invert:
        mat = torch.inverse(mat)
    return mat


def get_random_pose(range_u, range_v, range_radius, batch_size=32,
                    invert=False):
    loc = sample_on_sphere(range_u, range_v, size=(batch_size))
    radius = range_radius[0] + \
        torch.rand(batch_size) * (range_radius[1] - range_radius[0])
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def get_middle_pose(range_u, range_v, range_radius, batch_size=32,
                    invert=False):
    u_m, u_v, r_v = sum(range_u) * 0.5, sum(range_v) * \
        0.5, sum(range_radius) * 0.5
    loc = sample_on_sphere((u_m, u_m), (u_v, u_v), size=(batch_size))
    radius = torch.ones(batch_size) * r_v
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def get_camera_pose(range_u: object, range_v: object, range_r: object, val_u: object = 0.5, val_v: object = 0.5, val_r: object = 0.5,
                    batch_size: object = 32, invert: object = False) -> object:
    # range_u (tuple): rotation range (0 - 1)  高程和绕z轴的旋转二者构成了球坐标系，利用这二者求出在直角坐标系下的坐标，由于都是单位长度，所以要乘以参数r
    # range_v (tuple): elevation range (0 - 1)
    # r相当于一个射线长度
    # val_u这类值为[0,1]之间的幅度值
    u0, ur = range_u[0], range_u[1] - range_u[0]
    v0, vr = range_v[0], range_v[1] - range_v[0]
    r0, rr = range_r[0], range_r[1] - range_r[0]
    # 在每一个step下val_u等都是不同的值，故u等值也是不同的值，u即绕z轴旋转的角度，v即高度坐标z
    u = u0 + val_u * ur
    v = v0 + val_v * vr
    r = r0 + val_r * rr

    # 得到一个直角坐标系下的坐标 随机采样？得到的，单位长度
    loc = sample_on_sphere((u, u), (v, v), size=(batch_size))
    # print(loc.shape)   torch.Size([15, 3])

    radius = torch.ones(batch_size) * r
    loc = loc * radius.unsqueeze(-1)
    # print(loc.shape)  torch.Size([15, 3])
    R = look_at(loc)  # look_at函数返回的是r_mat旋转矩阵
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    # 旋转  RT尺寸为15*4*4  所有batch_size，0-2行，0-2列
    RT[:, :3, :3] = R
    # 相机位置 4*4  0-2行，-1指最后一列，即translation
    RT[:, :3, -1] = loc
    # print(RT[1].shape)  # torch.Size([4, 4])

    if invert:
        RT = torch.inverse(RT)
    return RT


def to_sphere(u, v):
    # range_u (tuple): rotation range (0 - 1)，0相当于0度，1相当于360度
    # range_v (tuple): elevation range (0 - 1)
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)  # ？没明白

    # 从球坐标转换到直角坐标系
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz], axis=-1)


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1), size=(1,),
                     to_pytorch=True):
    # 输入uv各自的范围，然后产生范围内的随机数，但其实get_camera_pose那里传来的range_u其实只包含一个数字
    # range_u (tuple): rotation range (0 - 1)
    # range_v (tuple): elevation range (0 - 1)
    # print(range_u)  在纯高程的情况下为(0.0, 0.0)
    # print(range_v)  在纯高程的情况下不断变化，但是里面也是两个相同的值，相当于单值
    u = np.random.uniform(*range_u, size=size)
    v = np.random.uniform(*range_v, size=size)
    # print(u) 纯高程下为[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    sample = to_sphere(u, v)
    if to_pytorch:
        sample = torch.tensor(sample).float()

    return sample


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5,
            to_pytorch=True):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)
    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis,
                                              axis=1, keepdims=True), eps]))

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis,
                                              axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis,
                                              axis=1, keepdims=True), eps]))

    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(
            -1, 3, 1)), axis=2)

    if to_pytorch:
        r_mat = torch.tensor(r_mat).float()

    return r_mat


def get_rotation_matrix(axis: object = 'z', value: object = 0., batch_size: object = 32) -> object:
    # 这里的batch_size会被外部的覆盖，我们默认采用的为16，即16种车型，对每一种车型（不同的shape和appearance）都进行一样的旋转
    r = Rot.from_euler(axis, value * 2 * np.pi).as_dcm()
    r = torch.from_numpy(r).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    return r
