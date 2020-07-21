import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    center_blue = (-3.0, -3.0, 'Cb')
    center_red = (+3.0, +3.0, 'Cr')
    centers = [center_red, center_blue]

    x = [center_blue[0], center_red[0]]
    y = [center_blue[1], center_red[1]]

    slope_ln = (y[0] - y[1]) / (x[0] - x[1])
    slope_ln_angle = np.degrees(np.arctan(slope_ln))

    w = [7.0, 10.0]
    prx = ((w[0] * x[0]) + (w[1] * x[1])) / sum(w)
    pry = ((w[0] * y[0]) + (w[1] * y[1])) / sum(w)
    pr = (prx, pry)

    w = [2.0, 2.0]
    pcenterx = ((w[0] * x[0]) + (w[1] * x[1])) / sum(w)
    pcentery = ((w[0] * y[0]) + (w[1] * y[1])) / sum(w)
    pcenter = (pcenterx, pcentery)

    slope_lp = - 1.0 / slope_ln
    offset_lp = pr[1] - (pr[0] * slope_lp)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # Mark centers
    mark_centers = True
    if mark_centers:
        cr = center_red
        cb = center_blue

        xshift = 0.5
        yshift = 0.5
        p = cr
        plt.scatter(p[0], p[1], marker='o', color='red', s=30, zorder=10)
        plt.text(p[0] + xshift, p[1] + yshift, '{} ({:>+.1f} ,{:>+.1f})'.format(p[2], p[0], p[1]),
                 fontsize=10, weight="bold")

        p = cb
        plt.scatter(p[0], p[1], marker='o', color='blue', s=30, zorder=10)
        plt.text(p[0] + xshift, p[1] - yshift, '{} ({:>+.1f} ,{:>+.1f})'.format(p[2], p[0], p[1]),
                 fontsize=10, weight="bold")

    # Plot L_n
    plot_straight_line = True
    if plot_straight_line:
        plt.plot(x, y, 'ro-', color='grey')

        w = [10.0, 1.0]
        px = (w[0] * x[0] + w[1] * x[1]) / sum(w)
        py = (w[0] * y[0] + w[1] * y[1]) / sum(w)
        p = (px, py)

        print "title: {}".format(p)
        print "rotation: {}".format(slope_ln_angle)
        plt.text(p[0], p[1] + 1, 'Ln',
                 fontsize=10, weight="bold", rotation=slope_ln_angle, )

    # Plot Pr (7:10 point) and center point
    plot_pr_line = True
    if plot_pr_line:
        plt.scatter(pr[0], pr[1], marker='o', color='black', s=30, zorder=10)
        plt.text(pr[0] + 0.8, pr[1], 'Pr',
                 fontsize=10, weight="bold", rotation=0)

        plt.scatter(pcenter[0], pcenter[1], marker='o', color='black', s=30, zorder=10)
        plt.text(pcenter[0] - 2.5, pcenter[1], 'Center',
                 fontsize=10, weight="bold", rotation=0)

    # Plot L_p
    plot_perpendicular_line = True
    if plot_perpendicular_line:
        x = np.linspace(-15, +15, 20)
        y = slope_lp * x + offset_lp

        xtitle = -5
        ytitle = slope_lp * xtitle + offset_lp
        tilt = np.degrees(np.arctan(slope_lp))

        plt.plot(x, y, 'r-', color='grey', linestyle='--')
        plt.text(xtitle + 0.2, ytitle, 'Lp',
                 fontsize=10, weight="bold", rotation=tilt, color='black')

    plt.gca().set_aspect("equal")
    # plt.grid()
    plt.tight_layout()
    plt.show()
