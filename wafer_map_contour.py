#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class WaferMapContour:
    """
    Wafer Map Contour - A powerful wafer map tool
    """
    @staticmethod
    def draw_map_contour(x: list, y: list, v: list, pic_name: str, wafer_size: int = 300, **kwargs: object) -> None:
        """ A Wafer Map Contour tool
        Draw and output wafer map file.
        :type kwargs: object
        :param x: 座標 X
        :param y: 座標 Y
        :param v: 座標 值
        :param pic_name: 圖片名稱
        :param wafer_size: Wafer 大小 (default = 300)
        :param kwargs:  vmin: 量測最小值 (default = z 最小值);
                        vmax: 量測最大值 (default = z 最大值);
                        output_path: 輸出圖檔的路徑(default = 預設路徑)
       """

        # Ensure data correct (fool-proofing)
        if not (len(x) == len(y) == len(v)):
            raise Exception('Input data size error!')

        # New layer
        fig, ax = plt.subplots()

        # find boundary coordinate & value
        boundary_x = [0, 0, (wafer_size / 2), -(wafer_size / 2)]
        boundary_y = [(wafer_size / 2), -(wafer_size / 2), 0, 0]
        xc = x.copy()
        yc = y.copy()
        vc = v.copy()
        for i in range(len(boundary_x)):
            xb = boundary_x[i]
            yb = boundary_y[i]
            distance = (wafer_size / 2)
            min_coordinate = list()
            # find minimize distance point, if plurality then get mean.
            for j in range(len(x)):
                d = ((xb - x[j]) ** 2 + (yb - y[j]) ** 2) ** (1 / 2)
                if d < distance:
                    min_coordinate = [j]
                    distance = d
                elif d == distance:
                    min_coordinate.append(j)
            min_value = [v[i] for i in min_coordinate]
            value = np.mean(min_value)
            xc.append(xb)
            yc.append(yb)
            vc.append(value)

        xc = np.array(xc)
        yc = np.array(yc)
        vc = np.array(vc)

        # Set up a regular grid of interpolation points
        xi, yi = np.linspace(xc.min(), xc.max(), 100), np.linspace(yc.min(), yc.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate
        rbf = scipy.interpolate.Rbf(xc, yc, vc, function='thin_plate')
        zi = rbf(xi, yi)

        vmin = kwargs['vmin'] if 'vmin' in kwargs else vc.min()
        vmax = kwargs['vmax'] if 'vmax' in kwargs else vc.max()

        im = ax.imshow(zi, vmin=vmin, vmax=vmax, origin='lower',
                       extent=[xc.min(), xc.max(), yc.min(), yc.max()], cmap='gist_rainbow_r',
                       )

        circle = Circle((0, 0), (wafer_size / 2), facecolor='none', linewidth=1)
        ax.add_patch(circle)
        im.set_clip_path(circle)
        fig.colorbar(im)

        xj = np.array(x)
        yj = np.array(y)
        vj = np.array(v)
        plt.plot(xj, yj, 'o', color='black', ms=5)

        for j in range(len(xj)):
            plt.text(xj[j], yj[j] + 7, str("{0:.2f}".format(round(vj[j], 2))), horizontalalignment='center',
                     verticalalignment='bottom')

        # Output picture
        output_path = kwargs['output_path'] if 'output_path' in kwargs else ''
        plt.savefig(os.path.join(output_path, pic_name))
        plt.close(fig)
