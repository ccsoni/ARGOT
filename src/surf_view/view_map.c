/*
 * view_map() : a rewrite of the PGPLOT version using matplotlib-c (libmplc.a)
 *
 * Each call produces one complete figure: it starts a Python process
 * (mplc_open), draws the map, saves it to filename, and closes the process
 * (mplc_close). The output format is determined by the file extension
 * (.pdf, .png, .eps, ...).
 *
 *   view_map(map, xscale, yscale, nbinx, nbiny,
 *            1, nbinx, 1, nbiny, vmin, vmax, "x", "y", "map.pdf");
 *
 * Returns 0 on success, -1 if the Python process could not be started.
 *
 * The memory layout of map is the same as in PGPLOT: map[ix + nbinx*iy]
 * (x is the fastest-varying index).
 * x_start..x_end and y_start..y_end are 1-based inclusive ranges,
 * as in cpgimag.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "matplotlib-c.h"

#define NCONT (10)

/*
 * Name of a matplotlib built-in colormap used for the image.
 * 'jet' is the closest to the PGPLOT rainbow table used in the original
 * code. Override at compile time, e.g. -DVIEW_MAP_CMAP=\"viridis\".
 */
#ifndef VIEW_MAP_CMAP
#define VIEW_MAP_CMAP "jet"
#endif

/* Python interpreter used by mplc_open (needs matplotlib and numpy) */
#ifndef VIEW_MAP_PYTHON
#define VIEW_MAP_PYTHON "python3"
#endif

/*
 * mplc device number used internally. If the caller also uses libmplc
 * directly, avoid opening this device number while view_map() is called.
 * Must be smaller than mplc_MaxDeviceNumber (16) in matplotlib-c.c.
 */
#ifndef VIEW_MAP_DEV
#define VIEW_MAP_DEV (0)
#endif

int view_map(float *map, float *xscale, float *yscale,
             int nbinx, int nbiny,
             int x_start, int x_end, int y_start, int y_end,
             float vmin, float vmax,
             char *xlabel, char *ylabel, const char *filename)
{
  const int dev = VIEW_MAP_DEV;
  int ix, iy;

  assert(filename != NULL && filename[0] != '\0');

  /* Automatic scaling when vmin == vmax (same behavior as the original) */
  if(vmin == vmax) {
    vmax = 0.0;
    vmin = 0.0;
    for(ix=0;ix<nbinx*nbiny;ix++) {
      if(vmax < map[ix]) vmax = map[ix];
    }
    vmax = ceil(vmax);
  }

  assert(1 <= x_start && x_start <= x_end && x_end <= nbinx);
  assert(1 <= y_start && y_start <= y_end && y_end <= nbiny);

  /* Extract the subarray and convert it to double
     (mplc_imshow takes a double*) */
  const int nx = x_end - x_start + 1;
  const int ny = y_end - y_start + 1;
  double *zsub = (double *)malloc(sizeof(double)*(size_t)nx*(size_t)ny);
  assert(zsub != NULL);

  for(iy=0;iy<ny;iy++) {
    for(ix=0;ix<nx;ix++) {
      zsub[ix + nx*iy] = map[(x_start-1+ix) + nbinx*(y_start-1+iy)];
    }
  }

  /* Axis ranges: xscale[x_start]..xscale[x_end-1], as in the original cpgaxis calls */
  const double xmin = xscale[x_start], xmax = xscale[x_end-1];
  const double ymin = yscale[y_start], ymax = yscale[y_end-1];

  /* Start a Python process for this figure */
  if(!mplc_open(dev, VIEW_MAP_PYTHON)) {
    fprintf(stderr, "view_map: failed to start %s\n", VIEW_MAP_PYTHON);
    free(zsub);
    return -1;
  }

  /* Figure setup: 5-inch-wide figure as in cpgpap(5.0, ...), and
     cpgscf(2) (Roman font). The margins are trimmed when saving
     (bbox_inches='tight'), so the height follows the data aspect ratio. */
  mplc_send_command(dev, "fig.set_size_inches(5.0, 5.25)");
  mplc_send_command(dev, "plt.rcParams['font.family'] = 'serif'");
  mplc_send_command(dev, "plt.rcParams['mathtext.fontset'] = 'dejavuserif'");

  /* Image (equivalent to cpgimag)
   * mplc_imshow does reshape((gridsize_x, gridsize_y)) on the Python side,
   * so (ny, nx) is passed in this order to match the C layout [iy][ix].
   * After the call, zvalues (2D array) and im remain on the Python side.
   * aspect='equal' draws x and y with the same scale (data units), so the
   * map is not stretched to fill the axes box. */
  char opt[512];
  snprintf(opt, sizeof(opt),
           "cmap='%s', vmin=%.9g, vmax=%.9g, "
           "extent=[%.9g, %.9g, %.9g, %.9g], "
           "interpolation='nearest', aspect='equal'",
           VIEW_MAP_CMAP, vmin, vmax, xmin, xmax, ymin, ymax);
  mplc_imshow(dev, zsub, ny, nx, opt, false, "");

  /* Contours (equivalent to cpgcont): NCONT levels evenly spaced in vmin..vmax
   * mplc_contour only supports square grids, so ax.contour is called directly
   * on zvalues left by mplc_imshow (this also avoids sending the data twice). */
  if(vmax > vmin) {
    mplc_send_command(dev,
                      "ax.contour(zvalues, levels=np.linspace(%.9g, %.9g, %d), "
                      "extent=[%.9g, %.9g, %.9g, %.9g], origin='lower', "
                      "colors='k', linewidths=0.5)",
                      vmin, vmax, NCONT, xmin, xmax, ymin, ymax);
  }

  /* Axis labels and ticks (equivalent to the inward ticks drawn on all four
   * sides with cpgaxis).
   * The labels are set with plt.xlabel, so this must be done before the
   * color-bar axes are created. */
  mplc_xlabel(dev, xlabel);
  mplc_ylabel(dev, ylabel);
  mplc_send_command(dev, "ax.tick_params(which='both', direction='in', top=True, right=True, labelfontfamily='serif')");
  mplc_send_command(dev, "ax.minorticks_on()");
  /* ax is created in mplc_open, so the rcParams change above does not
     apply to its existing labels */
  mplc_send_command(dev, "ax.xaxis.label.set_family('serif'); ax.yaxis.label.set_family('serif')");

  /* Color bar on the right of the map (as in the original).
   * make_axes_locatable keeps the bar the same height as the map even
   * though the axes box shrinks to the data aspect ratio. */
  mplc_send_command(dev, "from mpl_toolkits.axes_grid1 import make_axes_locatable");
  mplc_send_command(dev, "cax = make_axes_locatable(ax).append_axes('right', size='5%%', pad=0.1)");
  mplc_send_command(dev, "cb = fig.colorbar(im, cax=cax)");
  mplc_send_command(dev, "cb.ax.tick_params(which='both', direction='in')");
  mplc_send_command(dev, "plt.sca(ax)");

  /* Save and close; mplc_close waits for the Python process to finish,
     so the file is complete when view_map returns */
  mplc_save(dev, filename, "bbox_inches='tight'");
  mplc_close(dev);

  free(zsub);
  return 0;
}

#undef NCONT
