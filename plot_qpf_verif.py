#!/bin/usr/env python

import grib2io
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib
import io
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
import numpy as np
import time,os,sys,multiprocessing
import multiprocessing.pool
from scipy import ndimage
from scipy.ndimage.filters import minimum_filter, maximum_filter
from netCDF4 import Dataset
import pyproj
import cartopy
from datetime import datetime
import rrfs_plot_utils
import dateutil.relativedelta, dateutil.parser
from subprocess import call
from matplotlib import colors


####################################
# UTILITIES
####################################

def ndate(cdate,hours):
   if not isinstance(cdate, str):
     if isinstance(cdate, int):
       cdate=str(cdate)
     else:
       sys.exit('NDATE: Error - input cdate must be string or integer.  Exit!')
   if not isinstance(hours, int):
     if isinstance(hours, str):
       hours=int(hours)
     else:
       sys.exit('NDATE: Error - input delta hour must be a string or integer.  Exit!')
    
   indate=cdate.strip()
   hh=indate[8:10]
   yyyy=indate[0:4]
   mm=indate[4:6]
   dd=indate[6:8]
   #set date/time field
   parseme=(yyyy+' '+mm+' '+dd+' '+hh)
   datetime_cdate=dateutil.parser.parse(parseme)
   valid=datetime_cdate+dateutil.relativedelta.relativedelta(hours=+hours)
   vyyyy=str(valid.year)
   vm=str(valid.month).zfill(2)
   vd=str(valid.day).zfill(2)
   vh=str(valid.hour).zfill(2)
   return vyyyy+vm+vd+vh


def get_panel_spacing(dom, type='4panel'):
    if type in ['4panel']:
        if dom in ['conus']:
            wspace=0.1
            hspace=-0.7    
        elif dom in ['puerto_rico']:
            wspace=0.1
            hspace=-0.65    
        elif dom in ['sf_bay_area']:
            wspace=0.1
            hspace=-0.6    
        elif dom in ['south_central']:
            wspace=0.1
            hspace=-0.55    
        elif dom in ['boston_nyc']:
            wspace=0.1
            hspace=-0.5    
        elif dom in ['north_central','south_florida']:
            wspace=0.1
            hspace=-0.3    
        elif dom in ['southwest','northeast']:
            wspace=0.1
            hspace=-0.1    
        elif dom in [
                'alaska','hawaii','central','colorado','la_vegas','mid_atlantic',
                'northwest','ohio_valley','southeast','seattle_portland']:
            wspace=0.1
            hspace=0.1
        else:
            wspace=0.1
            hspace=0.0
    elif type in ['3panel']:
        if dom in ['conus']:
            wspace=0.1
            hspace=-0.3    
        elif dom in ['puerto_rico']:
            wspace=0.1
            hspace=-0.3    
        elif dom in ['sf_bay_area']:
            wspace=0.1
            hspace=-0.3    
        elif dom in ['south_central']:
            wspace=0.1
            hspace=-0.3    
        elif dom in ['boston_nyc']:
            wspace=0.1
            hspace=-0.3    
        elif dom in ['north_central','south_florida']:
            wspace=0.1
            hspace=-0.3    
        elif dom in [
                'alaska','hawaii','central','colorado','la_vegas','mid_atlantic',
                'northwest','ohio_valley','southeast','seattle_portland']:
            wspace=0.1
            hspace=-0.3
        elif dom in ['alaska']:
            wspace=0.1
            hspace=-0.3
        else:
            wspace=0.1
            hspace=0.0
    elif type in ['2panel']:
        if dom in ['conus']:
            wspace=0.1
            hspace=-0.7    
        elif dom in ['puerto_rico']:
            wspace=0.1
            hspace=-0.65    
        elif dom in ['sf_bay_area']:
            wspace=0.1
            hspace=-0.6    
        elif dom in ['south_central']:
            wspace=0.1
            hspace=-0.55    
        elif dom in ['boston_nyc']:
            wspace=0.1
            hspace=-0.5    
        elif dom in ['north_central','south_florida']:
            wspace=0.1
            hspace=-0.3    
        elif dom in ['southwest','northeast']:
            wspace=0.1
            hspace=-0.1    
        elif dom in [
                'alaska','hawaii','central','colorado','la_vegas','mid_atlantic',
                'northwest','ohio_valley','southeast','seattle_portland']:
            wspace=0.1
            hspace=0.1
        else:
            wspace=0.1
            hspace=0.0
    else:
        ValueError(f"Type \"{type}\" is not a valid plot type.")
    return wspace, hspace

def clear_plotables(ax,keep_ax_lst,fig):
  #### - step to clear off old plottables but leave the map info - ####
  if len(keep_ax_lst) == 0 :
    print("clear_plotables WARNING keep_ax_lst has length 0. Clearing ALL plottables including map info!")
  cur_ax_children = ax.get_children()[:]
  if len(cur_ax_children) > 0:
    for a in cur_ax_children:
      if a not in keep_ax_lst:
       # if the artist isn't part of the initial set up, remove it
        a.remove()

def convert_and_save(filename):
  #### - convert and save the image - ####
  plt.savefig(filename+'.png', bbox_inches='tight',dpi=150)
  #os.system('convert '+filename+'.png '+filename+'.gif')
  img = Image.open(filename + '.png')
  img.save(filename + '.gif', format='GIF')
  os.remove(filename+'.png')

def convert_and_save_2(filename):
  #### - convert and save the image - ####
  #### - use higher dpi for single panel plots - ####
  plt.savefig(filename+'.png', bbox_inches='tight',dpi=250)
  os.system('convert '+filename+'.png '+filename+'.gif')
  os.remove(filename+'.png')

def extrema(mat,mode='wrap',window=10):
  # From: http://matplotlib.org/basemap/users/examples.html
  
  """find the indices of local extrema (min and max)
  in the input array."""
  mn = minimum_filter(mat, size=window, mode=mode)
  mx = maximum_filter(mat, size=window, mode=mode)
  # (mat == mx) true if pixel is equal to the local max  # (mat == mn) true if pixel is equal to the local in
  # Return the indices of the maxima, minima
  return np.nonzero(mat == mn), np.nonzero(mat == mx)

def plt_highs_and_lows(x,y,mat,xmin,xmax,ymin,ymax,offset,ax,transform,mode='wrap',window=10):
  # From: http://matplotlib.org/basemap/users/examples.html
  if isinstance(window,int) == False:
    raise TypeError("The window argument to plt_highs_and_lows must be an integer.")
  local_min, local_max = extrema(mat,mode,window)
  xlows = x[local_min]; xhighs = x[local_max]
  ylows = y[local_min]; yhighs = y[local_max]
  lowvals = mat[local_min]; highvals = mat[local_max]
  # plot lows as red L's, with min pressure value underneath.
  xyplotted = []
  # don't plot if there is already a L or H within dmin meters.
#  yoffset = 0.022*(ymax-ymin)
  yoffset = offset
  dmin = yoffset
  for x,y,p in zip(xlows, ylows, lowvals):
    x_proj, y_proj = ax.projection.transform_point(x, y, ccrs.PlateCarree())
    if x_proj < xmax and x_proj > xmin and y_proj < ymax and y_proj > ymin:
#        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
#        if not dist or min(dist) > dmin:
            ax.text(x,y,'L',fontsize=14,fontweight='bold',
                    ha='center',va='center',color='r',zorder=4,clip_on=True,
                    transform=transform)
            ax.text(x,y-yoffset,repr(int(p)),fontsize=6,zorder=4,
                    ha='center',va='top',color='r',
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)),clip_on=True,
                    transform=transform)
            xyplotted.append((x,y))
  # plot highs as blue H's, with max pressure value underneath.
  xyplotted = []
  for x,y,p in zip(xhighs, yhighs, highvals):
    x_proj, y_proj = ax.projection.transform_point(x, y, ccrs.PlateCarree())
    if x_proj < xmax and x_proj > xmin and y_proj < ymax and y_proj > ymin:
#        dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
#        if not dist or min(dist) > dmin:
            ax.text(x,y,'H',fontsize=14,fontweight='bold',
                    ha='center',va='center',color='b',zorder=4,clip_on=True,
                    transform=transform)
            ax.text(x,y-yoffset,repr(int(p)),fontsize=6,
                    ha='center',va='top',color='b',zorder=4,
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)),clip_on=True,
                    transform=transform)
            xyplotted.append((x,y))

def get_latlons_pcolormesh(msg):
# Get shifted lats and lons for plotting with pcolormesh
  lats = []
  lons = []
  lats_shift = []
  lons_shift = []

# Unshifted grid for contours and wind barbs
  lat, lon = msg.grid()
  lats.append(lat)
  lons.append(lon)

# Shift grid for pcolormesh
  lat1 = msg.latitudeFirstGridpoint
  lon1 = msg.longitudeFirstGridpoint
  nx = msg.nx
  ny = msg.ny
  dx = msg.gridlengthXDirection
  dy = msg.gridlengthYDirection
  pj = pyproj.Proj(msg.projParameters)
  llcrnrx, llcrnry = pj(lon1,lat1)
  llcrnrx = llcrnrx - (dx/2.)
  llcrnry = llcrnry - (dy/2.)
  x = llcrnrx + dx*np.arange(nx)
  y = llcrnry + dy*np.arange(ny)
  x,y = np.meshgrid(x,y)
  lon, lat = pj(x, y, inverse=True)
  lats_shift.append(lat)
  lons_shift.append(lon)

# Unshifted lat/lon arrays grabbed directly using latlons() method
  lat = lats[0]
  lon = lons[0]

# Shifted lat/lon arrays for pcolormesh
  lat_shift = lats_shift[0]
  lon_shift = lons_shift[0]

# Fix for Alaska
  if (np.min(lon_shift) < 0) and (np.max(lon_shift) > 0):
    lon_shift = np.where(lon_shift>0,lon_shift-360,lon_shift)

  return lat, lon, lat_shift, lon_shift


####################################
#  Color shading / Color bars
####################################

def cmap_t2m():
 # Create colormap for 2-m temperature
 # Modified version of the ncl_t2m colormap from Jacob's ncepy code
    r=np.array([255,128,0,  70, 51, 0,  255,0, 0,  51, 255,255,255,255,255,171,128,128,36,162,255])
    g=np.array([0,  0,  0,  70, 102,162,255,92,128,185,255,214,153,102,0,  0,  0,  68, 36,162,255])
    b=np.array([255,128,128,255,255,255,255,0, 0,  102,0,  112,0,  0,  0,  56, 0,  68, 36,162,255])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_t2m_coltbl = colors.LinearSegmentedColormap('CMAP_T2M_COLTBL',colorDict)
    return cmap_t2m_coltbl

def cmap_q2m():
  # Create colormap for dew point temperature
    r=np.array([255,179,96,128,0, 0,  51, 0,  0,  0,  133,51, 70, 0,  128,128,180])
    g=np.array([255,179,96,128,92,128,153,155,155,255,162,102,70, 0,  0,  0,  0])
    b=np.array([255,179,96,0,  0, 0,  102,155,255,255,255,255,255,128,255,128,128])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_q2m_coltbl = colors.LinearSegmentedColormap('CMAP_Q2M_COLTBL',colorDict)
    cmap_q2m_coltbl.set_over(color='deeppink')
    return cmap_q2m_coltbl

def cmap_t850():
 # Create colormap for 850-mb equivalent potential temperature
    r=np.array([255,128,0,  70, 51, 0,  0,  0, 51, 255,255,255,255,255,171,128,128,96,201])
    g=np.array([0,  0,  0,  70, 102,162,225,92,153,255,214,153,102,0,  0,  0,  68, 96,201])
    b=np.array([255,128,128,255,255,255,162,0, 102,0,  112,0,  0,  0,  56, 0,  68, 96,201])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_t850_coltbl = colors.LinearSegmentedColormap('CMAP_T850_COLTBL',colorDict)
    return cmap_t850_coltbl

def cmap_terra():
 # Create colormap for terrain height
 # Emerald green to light green to tan to gold to dark red to brown to light brown to white
    r=np.array([0,  152,212,188,127,119,186])
    g=np.array([128,201,208,148,34, 83, 186])
    b=np.array([64, 152,140,0,  34, 64, 186])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b=np.array([255,179,96,0,  0, 0,  102,155,255,255,255,255,255,128,255,128,128])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_q2m_coltbl = colors.LinearSegmentedColormap('CMAP_Q2M_COLTBL',colorDict)
    cmap_q2m_coltbl.set_over(color='deeppink')
    return cmap_q2m_coltbl

def cmap_t850():
 # Create colormap for 850-mb equivalent potential temperature
    r=np.array([255,128,0,  70, 51, 0,  0,  0, 51, 255,255,255,255,255,171,128,128,96,201])
    g=np.array([0,  0,  0,  70, 102,162,225,92,153,255,214,153,102,0,  0,  0,  68, 96,201])
    b=np.array([255,128,128,255,255,255,162,0, 102,0,  112,0,  0,  0,  56, 0,  68, 96,201])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_t850_coltbl = colors.LinearSegmentedColormap('CMAP_T850_COLTBL',colorDict)
    return cmap_t850_coltbl

def cmap_terra():
 # Create colormap for terrain height
 # Emerald green to light green to tan to gold to dark red to brown to light brown to white
    r=np.array([0,  152,212,188,127,119,186])
    g=np.array([128,201,208,148,34, 83, 186])
    b=np.array([64, 152,140,0,  34, 64, 186])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_terra_coltbl = colors.LinearSegmentedColormap('CMAP_TERRA_COLTBL',colorDict)
    cmap_terra_coltbl.set_over(color='#E0EEE0')
    return cmap_terra_coltbl

def ncl_perc_11Lev():
  # Create colormap for snowfall
    r=np.array([202,89,139,96,26,145,217,254,252,215,150])
    g=np.array([202,141,239,207,152,207,239,224,141,48,0])
    b=np.array([200,252,217,145,80,96,139,139,89,39,100])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    blue = []
    green = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    my_coltbl = colors.LinearSegmentedColormap('NCL_PERC_11LEV_COLTBL',colorDict)
    return my_coltbl

def ncl_grnd_hflux():
  # Create colormap for ground heat flux
    r=np.array([0,8,16,24,32,40,48,85,133,181,230,253,253,253,253,253,253,253,253,253,253,
253])
    g=np.array([253,222,189,157,125,93,60,85,133,181,230,230,181,133,85,60,93,125,157,189,
224,253])
    b=np.array([253,253,253,253,253,253,253,253,253,253,253,230,181,133,85,48,40,32,24,16,
8,0])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    blue = []
    green = []
    for i in range(len(xsize)):
        xNorm=np.float(i)/(np.float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    my_coltbl = colors.LinearSegmentedColormap('NCL_GRND_HFLUX_COLTBL',colorDict)
    return my_coltbl


def domain_latlons_proj(dom):
#  These are the available pre-defined domains:
#  
#  namerica, caribbean, alaska, hawaii, puerto rico,
#  conus, northeast, mid_atlantic, southeast, ohio_valley,
#  upper_midwest, north_central, central, south_central,
#  northwest, southwest, colorado, boston_nyc,
#  seattle_portland, sf_bay_area, la_vegas

# Latitudes and longitudes
  if dom == 'namerica': 
    llcrnrlon = -160.0
    llcrnrlat = 15.0
    urcrnrlon = -55.0
    urcrnrlat = 65.0
    cen_lat = 35.4
    cen_lon = -105.0
    xextent = -3700000
    yextent = -2500000
    offset = 1
  elif dom == 'caribbean':
    llcrnrlon = -89.0
    llcrnrlat = 15.1
    urcrnrlon = -60.5
    urcrnrlat = 29.5
    cen_lat = 25.0
    cen_lon = -82.0
    xextent=-300000
    yextent=-725000
    offset=0.25
  elif dom == 'alaska':
    llcrnrlon = -167.5
    llcrnrlat = 50.5
    urcrnrlon = -135.8
    urcrnrlat = 72.5
    cen_lat = 60.0
    cen_lon = -150.0
    lat_ts = 60.0
    xextent=-850000
    yextent=-600000
    offset=1
  elif dom == 'hawaii':
    llcrnrlon = -162.3
    llcrnrlat = 16.2
    urcrnrlon = -153.1
    urcrnrlat = 24.3
    cen_lat = 20.4
    cen_lon = -157.6
    xextent=-325000
    yextent=-285000
    offset=0.25
  elif dom == 'puerto_rico':
    llcrnrlon = -76.5
    llcrnrlat = 13.3
    urcrnrlon = -61.0
    urcrnrlat = 22.7
    cen_lat = 18.4
    cen_lon = -66.6
    xextent=-925000
    yextent=-375000
    offset=0.25
  elif dom == 'conus':
    llcrnrlon = -125.5
    llcrnrlat = 20.0
    urcrnrlon = -63.5
    urcrnrlat = 51.0
    cen_lat = 35.4
    cen_lon = -97.6
    xextent=-2200000
    yextent=-675000
    offset=1
  elif dom == 'northeast':
    llcrnrlon = -80.0
    llcrnrlat = 40.0
    urcrnrlon = -66.5
    urcrnrlat = 48.0
    cen_lat = 44.0
    cen_lon = -76.0
    xextent=-175000
    yextent=-282791
    offset=0.25
  elif dom == 'mid_atlantic':
    llcrnrlon = -82.0
    llcrnrlat = 36.5
    urcrnrlon = -73.0
    urcrnrlat = 42.5
    cen_lat = 36.5
    cen_lon = -79.0
    xextent=-123114
    yextent=125850
    offset=0.25
  elif dom == 'southeast':
    llcrnrlon = -92.0
    llcrnrlat = 24.0
    urcrnrlon = -75.0
    urcrnrlat = 37.0
    cen_lat = 30.5
    cen_lon = -89.0
    xextent=-12438
    yextent=-448648
    offset=0.25
  elif dom == 'south_florida':
    llcrnrlon = -84.0
    llcrnrlat = 23.0
    urcrnrlon = -77
    urcrnrlat = 28.0
    cen_lat = 24.0
    cen_lon = -81.0
    xextent=-206000
    yextent=-6000
    offset=0.25
  elif dom == 'ohio_valley':
    llcrnrlon = -91.5
    llcrnrlat = 34.5
    urcrnrlon = -80.0
    urcrnrlat = 43.0
    cen_lat = 38.75
    cen_lon = -88.0
    xextent=-131129
    yextent=-299910
    offset=0.25
  elif dom == 'upper_midwest':
    llcrnrlon = -97.5
    llcrnrlat = 40.0
    urcrnrlon = -82.0
    urcrnrlat = 49.5
    cen_lat = 44.75
    cen_lon = -92.0
    xextent=-230258
    yextent=-316762
    offset=0.25
  elif dom == 'north_central':
    llcrnrlon = -111.5
    llcrnrlat = 39.0
    urcrnrlon = -94.0
    urcrnrlat = 49.5
    cen_lat = 44.25
    cen_lon = -103.0
    xextent=-490381
    yextent=-336700
    offset=0.25
  elif dom == 'central':
    llcrnrlon = -103.5
    llcrnrlat = 32.0
    urcrnrlon = -89.0
    urcrnrlat = 42.0
    cen_lat = 37.0
    cen_lon = -99.0
    xextent=-220257
    yextent=-337668
    offset=0.25
  elif dom == 'south_central':
    llcrnrlon = -109.0
    llcrnrlat = 25.0
    urcrnrlon = -88.5
    urcrnrlat = 37.5
    cen_lat = 31.25
    cen_lon = -101.0
    xextent=-529631
    yextent=-407090
    offset=0.25
  elif dom == 'northwest':
    llcrnrlon = -125.0
    llcrnrlat = 40.0
    urcrnrlon = -110.0
    urcrnrlat = 50.0
    cen_lat = 45.0
    cen_lon = -116.0
    xextent=-540000
    yextent=-333623
    offset=0.25
  elif dom == 'southwest':
    llcrnrlon = -125.0
    llcrnrlat = 31.0
    urcrnrlon = -108.5
    urcrnrlat = 42.5
    cen_lat = 36.75
    cen_lon = -116.0
    xextent=-593059
    yextent=-377213
    offset=0.25
  elif dom == 'colorado':
    llcrnrlon = -110.0
    llcrnrlat = 35.0
    urcrnrlon = -101.0
    urcrnrlat = 42.0
    cen_lat = 38.5
    cen_lon = -106.0
    xextent=-224751
    yextent=-238851
    offset=0.25
  elif dom == 'boston_nyc':
    llcrnrlon = -75.5
    llcrnrlat = 40.0
    urcrnrlon = -69.5
    urcrnrlat = 43.0
    cen_lat = 41.5
    cen_lon = -76.0
    xextent=112182
    yextent=-99031
    offset=0.25
  elif dom == 'seattle_portland':
    llcrnrlon = -126.0
    llcrnrlat = 44.5
    urcrnrlon = -118.0
    urcrnrlat = 49.5
    cen_lat = 47.0
    cen_lon = -121.0
    xextent=-275000
    yextent=-180000
    offset=0.25
  elif dom == 'sf_bay_area':
    llcrnrlon = -123.5
    llcrnrlat = 37.25
    urcrnrlon = -121.0
    urcrnrlat = 38.5
    cen_lat = 48.25
    cen_lon = -121.0
    xextent=-185364
    yextent=-1193027
    offset=0.25
  elif dom == 'la_vegas':
    llcrnrlon = -121.0
    llcrnrlat = 32.0
    urcrnrlon = -114.0
    urcrnrlat = 37.0
    cen_lat = 34.5
    cen_lon = -114.0
    xextent=-540000
    yextent=-173241
    offset=0.25

# Projection settings
  if dom == 'namerica':
    extent = [-176.,0.,0.5,45.]
    myproj = ccrs.Orthographic(central_longitude=-114, central_latitude=54.0, globe=None)
  elif dom == 'alaska':
    extent = [llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat]
    myproj = ccrs.Stereographic(central_longitude=cen_lon, central_latitude=cen_lat,
         true_scale_latitude=None,false_easting=0.0,false_northing=0.0,globe=None)
  elif dom == 'conus':
    extent = [llcrnrlon-1, urcrnrlon-6, llcrnrlat, urcrnrlat+1]
    myproj=ccrs.LambertConformal(central_longitude=cen_lon, central_latitude=cen_lat,
         false_easting=0.0, false_northing=0.0, secant_latitudes=None,
         standard_parallels=None, globe=None)
  else:
    extent = [llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat]
    myproj=ccrs.LambertConformal(central_longitude=cen_lon, central_latitude=cen_lat,
         false_easting=0.0, false_northing=0.0, secant_latitudes=None,
         standard_parallels=None, globe=None)

  return xextent, yextent, offset, extent, myproj

#-------------------------------------------------------#

# Necessary to generate figs when not running an Xserver (e.g. via PBS)
plt.switch_backend('agg')

# Read date/time and forecast hour from command line
ymdh = str(sys.argv[1])
ymd = ymdh[0:8]
year = int(ymdh[0:4])
month = int(ymdh[4:6])
day = int(ymdh[6:8])
hour = int(ymdh[8:10])
cyc = str(hour).zfill(2)
print(year, month, day, hour)

ymdh_model = str(sys.argv[2])
ymd_model = ymdh_model[0:8]
year_model = int(ymdh_model[0:4])
month_model = int(ymdh_model[4:6])
day_model = int(ymdh_model[6:8])
hour_model = int(ymdh_model[8:10])
cyc_model = str(hour_model).zfill(2)
print(year_model, month_model, day_model, hour_model)

fhr = int(sys.argv[3])
fhour = str(fhr).zfill(2)
fhourm24 = str(fhr-24).zfill(2)
print('fhour '+fhour)

# Forecast valid date/time
itime = ymd_model
vtime_start = ndate(ymdh_model,int(fhr-24))
vtime_start = str(vtime_start[0:8])
vtime_end = ymd

# Define the directory paths to the output files
STAGE_DIR = '/lfs/h2/emc/stmp/'+os.environ['USER']+'/rrfs_verif'
PARM_DIR = os.path.join(os.environ['HOMEDIR'],'parm')
COMccpa = os.environ['COMobs']
DCOMmrms = os.environ['COMobs']

HRRR_DIR = os.path.join(os.environ['COMfcst'],'hrrr.'+ymd_model)
NAM_DIR = os.path.join(os.environ['COMfcst'],'nam.'+ymd_model)
RRFS_DIR = os.path.join(
    '/','lfs','h2','emc','ptmp',os.environ['USER'],'rrfs','na','prod',
    'rrfs.'+ymd_model, cyc_model
)
CCPA_DIR = os.path.join(STAGE_DIR)
MRMS_DIR = os.path.join(STAGE_DIR)

# Set up working directories
if not os.path.exists(os.path.join(CCPA_DIR, 'tmp')):
    if not os.path.exists(CCPA_DIR):
        os.makedirs(CCPA_DIR)
    os.makedirs(os.path.join(CCPA_DIR, 'tmp'))
    os.makedirs(os.path.join(CCPA_DIR, 'logs'))
if not os.path.exists(os.path.join(MRMS_DIR, 'tmp')):
    if not os.path.exists(MRMS_DIR):
        os.makedirs(MRMS_DIR)
    os.makedirs(os.path.join(MRMS_DIR, 'tmp'))
    os.makedirs(os.path.join(MRMS_DIR, 'logs'))

# Specify plotting domains
domains = ['conus','alaska','hawaii','puerto_rico','boston_nyc','central','colorado','la_vegas','mid_atlantic','north_central','northeast','northwest','ohio_valley','south_central','southeast','south_florida','sf_bay_area','seattle_portland','southwest','upper_midwest']

# Paths to image files
user = str(sys.argv[4])
im = image.imread('/lfs/h2/emc/vpppg/noscrub/'+user+'/RRFS_eval_retro_graphics/noaa.png')

#-------------------------------------------------------#

# Make Python process pools non-daemonic
class NoDaemonProcess(multiprocessing.Process):
  # make 'daemon' attribute always return False
  @property
  def daemon(self):
    return False
  
  @daemon.setter
  def daemon(self, value):
    pass

class NoDaemonContext(type(multiprocessing.get_context())):
  Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
  def __init__(self, *args, **kwargs):
    kwargs['context'] = NoDaemonContext()
    super(MyPool, self).__init__(*args, **kwargs)

#-------------------------------------------------------#

def main():

  # Number of processes must coincide with the number of domains to plot
  pool = MyPool(len(domains))
  pool.map(vars_figure,domains)

#-------------------------------------------------------#

def vars_figure(domain):

  global dom
  dom = domain
  print(('Working on '+dom))

  global lat,lon,lat_shift,lon_shift,fig,axes,ax1,ax2,ax3,ax4,keep_ax_lst_1,keep_ax_lst_2,keep_ax_lst_3,xextent,yextent,offset,extent,myproj,transform

# Define the input files
  if dom == 'alaska':
      dom1a_string = 'ak.'
      dom1a_string2 = dom
      dom1b_string = 'awp242'
      dom2_string = 'alaska'
      dom3_string = 'ak'
      dom5_string = 'ak'
  elif dom == 'puerto_rico':
      dom1a_string = ''
      dom1a_string2 = dom
      dom1b_string = 'awp237'
      dom2_string = 'prico'
      dom3_string = 'pr'
      dom5_string = 'pr'
  elif dom == 'hawaii':
      dom1a_string = ''
      dom1a_string2 = dom
      dom1b_string = 'awiphi'
      dom2_string = 'hawaii'
      dom3_string = 'hi'
      dom5_string = 'hi'
  else:
      dom1a_string = ''
      dom1a_string2 = 'conus'
      dom1b_string = 'awip12'
      dom2_string = 'conus'
      dom3_string = 'conus'
      dom5_string = 'conus'
  
  if dom in ['alaska', 'puerto_rico', 'hawaii']: 
    use_ccpa = False
  else:
    use_ccpa = True
  
  fhour_03 = str(fhr - 21).zfill(2)
  fhour_06 = str(fhr - 18).zfill(2)
  fhour_09 = str(fhr - 15).zfill(2)
  fhour_12 = str(fhr - 12).zfill(2)
  fhour_15 = str(fhr - 9).zfill(2)
  fhour_18 = str(fhr - 6).zfill(2)
  fhour_21 = str(fhr - 3).zfill(2)
 
  plot_nodata_text = [False, False, False, False, False, False]
  fname1a = HRRR_DIR+f'/{dom1a_string2}/hrrr.t'+cyc_model+'z.wrfprsf'+fhour+f'.{dom1a_string}grib2'
  fname1a_fm24 = HRRR_DIR+f'/{dom1a_string2}/hrrr.t'+cyc_model+'z.wrfprsf'+fhourm24+f'.{dom1a_string}grib2'
  fname1b_03 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_03+'.tm00.grib2'
  fname1b_06 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_06+'.tm00.grib2'
  fname1b_09 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_09+'.tm00.grib2'
  fname1b_12 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_12+'.tm00.grib2'
  fname1b_15 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_15+'.tm00.grib2'
  fname1b_18 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_18+'.tm00.grib2'
  fname1b_21 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour_21+'.tm00.grib2'
  fname1b_24 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom1b_string}'+fhour+'.tm00.grib2'
  fname2_03 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_03+'.tm00.grib2'
  fname2_06 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_06+'.tm00.grib2'
  fname2_09 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_09+'.tm00.grib2'
  fname2_12 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_12+'.tm00.grib2'
  fname2_15 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_15+'.tm00.grib2'
  fname2_18 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_18+'.tm00.grib2'
  fname2_21 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour_21+'.tm00.grib2'
  fname2_24 = NAM_DIR+'/nam.t'+cyc_model+f'z.{dom2_string}nest.hiresf'+fhour+'.tm00.grib2'
  fname3 = RRFS_DIR+'/rrfs.t'+cyc_model+'z.prslev.f0'+fhour+f'.{dom3_string}.grib2'
  fname3_fm24 = RRFS_DIR+'/rrfs.t'+cyc_model+'z.prslev.f0'+fhourm24+f'.{dom3_string}.grib2'
  if use_ccpa:
      fname5 = CCPA_DIR+f'/ccpa.{ymd}/ccpa.t'+cyc+f'z.a24h.{dom5_string}.nc'
  else:
      fname5 = MRMS_DIR+f'/mrms.{ymd}/mrms.t'+cyc+f'z.a24h.{dom}.nc'
 
  for fname in [
          fname1a, fname1a_fm24
          ]:
      if not os.path.exists(fname):
          plot_nodata_text[0] = True
          break
  if not plot_nodata_text[0]:
      data1a = grib2io.open(fname1a)
      data1a_fm24 = grib2io.open(fname1a_fm24)
  for fname1b in [
          fname1b_03, fname1b_06, fname1b_09, fname1b_12, 
          fname1b_15, fname1b_18, fname1b_21, fname1b_24
          ]:
      if not os.path.exists(fname1b):
          plot_nodata_text[1] = True
          break
  if not plot_nodata_text[1]:
      data1b_03 = grib2io.open(fname1b_03)
      data1b_06 = grib2io.open(fname1b_06)
      data1b_09 = grib2io.open(fname1b_09)
      data1b_12 = grib2io.open(fname1b_12)
      data1b_15 = grib2io.open(fname1b_15)
      data1b_18 = grib2io.open(fname1b_18)
      data1b_21 = grib2io.open(fname1b_21)
      data1b_24 = grib2io.open(fname1b_24)
  for fname2 in [
          fname2_03, fname2_06, fname2_09, fname2_12, 
          fname2_15, fname2_18, fname2_21, fname2_24]:
      if not os.path.exists(fname2):
          plot_nodata_text[2] = True
          break
  if not plot_nodata_text[2]:
      data2_03 = grib2io.open(fname2_03)
      data2_06 = grib2io.open(fname2_06)
      data2_09 = grib2io.open(fname2_09)
      data2_12 = grib2io.open(fname2_12)
      data2_15 = grib2io.open(fname2_15)
      data2_18 = grib2io.open(fname2_18)
      data2_21 = grib2io.open(fname2_21)
      data2_24 = grib2io.open(fname2_24)
  for fname in [
          fname3, fname3_fm24
          ]:
      if not os.path.exists(fname):
          plot_nodata_text[3] = True
          print(
              f"WARNING: No RRFS file found for VDATE={cyc}Z {ymd} at F{fhour}: {fname}"
          )
          break
  if not plot_nodata_text[3]:
      print(f"File exists: {fname3}. Attempting to open!")
      try:
        data3 = grib2io.open(fname3)
      except KeyError:
        plot_nodata_text[3] = True
      print(f"File exists: {fname3_fm24}. Attempting to open!")
      try:
        data3_fm24 = grib2io.open(fname3_fm24)
      except KeyError:
        plot_nodata_text[3] = True
  if os.path.exists(fname5):
      data5 = Dataset(fname5,'r')
  else:
      plot_nodata_text[5] = True

# Get the lats and lons
  if not plot_nodata_text[0]:
    msg = data1a.select(shortName='HGT', level='500 mb')[0]  # msg is a Grib2Message object
    lat1a,lon1a,lat1a_shift,lon1a_shift = get_latlons_pcolormesh(msg)
  if not plot_nodata_text[1]:
    msg = data1b_24.select(shortName='HGT', level='surface')[0]  # msg is a Grib2Message object
    lat1b,lon1b,lat1b_shift,lon1b_shift = get_latlons_pcolormesh(msg)
  if not plot_nodata_text[2]:
    msg = data2_24.select(shortName='HGT', level='500 mb')[0]  # msg is a Grib2Message object
    lat2,lon2,lat2_shift,lon2_shift = get_latlons_pcolormesh(msg)
  if not plot_nodata_text[3]:
    msg = data3.select(shortName='HGT', level='500 mb')[0]  # msg is a Grib2Message object
    lat3,lon3,lat3_shift,lon3_shift = get_latlons_pcolormesh(msg)
# CCPA/MRMS
  if not plot_nodata_text[5]:
    if dom in ['alaska','hawaii','puerto_rico']:
        lat5 = data5.variables['lat'][:]
        lon5 = data5.variables['lon'][:]
        lon5, lat5 = np.meshgrid(lon5, lat5)
    else:
        lat5 = data5.variables['lat'][:,:]
        lon5 = data5.variables['lon'][:,:]

###################################################
# Read in all variables and calculate differences #
###################################################
  t1a = time.perf_counter()

  global qpf_1a,qpf_1b,qpf_2,qpf_3,qpf_5

# Total Precipitation
  if not plot_nodata_text[0]:
      qpf_1a_f = data1a.select(shortName='APCP',timeRangeOfStatisticalProcess=fhr)[0].data * 0.0393701
      qpf_1a_fm24 = data1a_fm24.select(shortName='APCP',timeRangeOfStatisticalProcess=(fhr-24))[0].data * 0.0393701
      qpf_1a = qpf_1a_f - qpf_1a_fm24
  if not plot_nodata_text[1]:
      qpf_1b_03 = data1b_03.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_1b_06 = data1b_06.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_1b_09 = data1b_09.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_1b_12 = data1b_12.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_1b_15 = data1b_15.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_1b_18 = data1b_18.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_1b_21 = data1b_21.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_1b_24 = data1b_24.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_1b = qpf_1b_03 + qpf_1b_06 + qpf_1b_09 + qpf_1b_12 + qpf_1b_15 + qpf_1b_18 + qpf_1b_21 + qpf_1b_24
  if not plot_nodata_text[2]:
      qpf_2_03 = data2_03.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_06 = data2_06.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_09 = data2_09.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_12 = data2_12.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_15 = data2_15.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_18 = data2_18.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_21 = data2_21.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2_24 = data2_24.select(shortName='APCP',timeRangeOfStatisticalProcess=3)[0].data * 0.0393701
      qpf_2 = qpf_2_03 + qpf_2_06 + qpf_2_09 + qpf_2_12 + qpf_2_15 + qpf_2_18 + qpf_2_21 + qpf_2_24
  if not plot_nodata_text[3]:
      qpf_3_f = data3.select(shortName='APCP')[1].data * 0.0393701
      if fhr > 24:
        qpf_3_fm24 = data3_fm24.select(shortName='APCP')[1].data * 0.0393701
        qpf_3 = qpf_3_f - qpf_3_fm24
      else:
        qpf_3 = qpf_3_f
  if not plot_nodata_text[5]:
    # CCPA/MRMS
    if dom in ['alaska','hawaii','puerto_rico']:
      qpf_5 = data5.variables['MultiSensor_QPE_24H_Pass2_Z0'][:,:] * 0.0393701
    else:
      qpf_5 = data5.variables['APCP_24'][:,:] * 0.0393701

  t2a = time.perf_counter()
  t3a = round(t2a-t1a, 3)
  print(("%.3f seconds to read all messages") % t3a)

#######################################
#    SET UP FIGURE FOR EACH DOMAIN    #
#######################################

# Call the domain_latlons_proj function from rrfs_plot_utils
  xextent,yextent,offset,extent,myproj = domain_latlons_proj(dom)

#######################################
#  RUN 4-PANEL FOR MODELS 1A AND 1B   #
#######################################

  for use_mod1 in ['a','b']:
    # Select data based on which model 1 is used
      if use_mod1 == 'a': 
        if not plot_nodata_text[0]:
          use_qpf1 = qpf_1a
          use_lon1_shift = lon1a_shift
          use_lat1_shift = lat1a_shift
          if dom in ['puerto_rico', 'hawaii']:
              plot_nodata_text[0] = True
        mod1_name = 'HRRR'
      elif use_mod1 == 'b':
        if not plot_nodata_text[1]:
          use_qpf1 = qpf_1b
          use_lon1_shift = lon1b_shift
          use_lat1_shift = lat1b_shift
        mod1_name = 'NAM'
      else:
          raise ValueError(
              f'Unrecognized option for use_mod1: {use_mod1}'
          )    

    # Create figure and axes instances
      fig = plt.figure(figsize=(8,8))           
      ws, hs = get_panel_spacing(dom)
      gs = GridSpec(8,8,wspace=ws,hspace=hs)

      # Define where Cartopy maps are located
      cartopy.config['data_dir'] = '/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth'
      back_res='50m'
      back_img='off'

      ax1 = fig.add_subplot(gs[0:4,0:4], projection=myproj)
      ax2 = fig.add_subplot(gs[0:4,4:], projection=myproj)
      ax3 = fig.add_subplot(gs[4:,0:4], projection=myproj)
      ax4 = fig.add_subplot(gs[4:,4:], projection=myproj)
      ax1.set_extent(extent)
      ax2.set_extent(extent)
      ax3.set_extent(extent)
      ax4.set_extent(extent)
      axes = [ax1, ax2, ax3, ax4]

      fline_wd = 0.5  # line width
      fline_wd_lakes = 0.25  # line width
      falpha = 0.5    # transparency

      # natural_earth
      lakes=cfeature.NaturalEarthFeature('physical','lakes',back_res,
                        edgecolor='black',facecolor='none',
                        linewidth=fline_wd_lakes)
      coastline=cfeature.NaturalEarthFeature('physical','coastline',
                        back_res,edgecolor='black',facecolor='none',
                        linewidth=fline_wd,alpha=falpha)
      states=cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces',
                        back_res,edgecolor='black',facecolor='none',
                        linewidth=fline_wd,alpha=falpha)

      # All lat lons are earth relative, so setup the associated projection correct for that data
      transform = ccrs.PlateCarree()

      # high-resolution background images
      if back_img=='on':
         img = plt.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth/raster_files/NE1_50M_SR_W.tif')
         ax1.imshow(img, origin='upper', transform=transform)
         ax2.imshow(img, origin='upper', transform=transform)
         ax3.imshow(img, origin='upper', transform=transform)
         ax4.imshow(img, origin='upper', transform=transform)

      ax1.add_feature(cfeature.LAND, linewidth=0, facecolor='white')
      ax1.add_feature(cfeature.OCEAN, linewidth=0, facecolor='lightgray')
      ax1.add_feature(cfeature.LAKES, edgecolor='black', linewidth=fline_wd_lakes, facecolor='lightgray',zorder=0)
      ax1.add_feature(lakes)
      ax1.add_feature(states)
      ax1.add_feature(coastline)
      ax2.add_feature(cfeature.LAND, linewidth=0, facecolor='white')
      ax2.add_feature(cfeature.OCEAN, linewidth=0, facecolor='lightgray')
      ax2.add_feature(cfeature.LAKES, edgecolor='black', linewidth=fline_wd_lakes, facecolor='lightgray',zorder=0)
      ax2.add_feature(lakes)
      ax2.add_feature(states)
      ax2.add_feature(coastline)
      ax3.add_feature(cfeature.LAND, linewidth=0, facecolor='white')
      ax3.add_feature(cfeature.OCEAN, linewidth=0, facecolor='lightgray')
      ax3.add_feature(cfeature.LAKES, edgecolor='black', linewidth=fline_wd_lakes, facecolor='lightgray',zorder=0)
      ax3.add_feature(lakes)
      ax3.add_feature(states)
      ax3.add_feature(coastline)
      ax4.add_feature(cfeature.LAND, linewidth=0, facecolor='white')
      ax4.add_feature(cfeature.OCEAN, linewidth=0, facecolor='lightgray')
      ax4.add_feature(cfeature.LAKES, edgecolor='black', linewidth=fline_wd_lakes, facecolor='lightgray',zorder=0)
      ax4.add_feature(lakes)
      ax4.add_feature(states)
      ax4.add_feature(coastline)

      # Map/figure has been set up here, save axes instances for use again later
      keep_ax_lst_1 = ax1.get_children()[:]
      keep_ax_lst_2 = ax2.get_children()[:]
      keep_ax_lst_3 = ax3.get_children()[:]
      keep_ax_lst_4 = ax4.get_children()[:]

      xmin, xmax = ax1.get_xlim()
      ymin, ymax = ax1.get_ylim()
      xmax = int(round(xmax))
      ymax = int(round(ymax))

    #################################
      # Plot 24-hr QPF
    #################################
      datasets = ['CCPA-MRMS']
      for pcpanl in datasets:

        t1 = time.perf_counter()
        print((
            'Working on 24-hr QPF for '+dom
            +(' with HRRR' if use_mod1=='a' else (
                ' with NAM' if use_mod1=='b' 
                else ' with ~mystery model~'
            ))
        ))

        units = 'Precipitation (inches)'
        clevs = [0.01,0.1,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,7,10,15,20]
        colorlist = ['chartreuse','limegreen','green','blue','dodgerblue','deepskyblue','cyan','mediumpurple','mediumorchid','darkmagenta','darkred','crimson','orangered','darkorange','goldenrod','gold','yellow']  
        cm = matplotlib.colors.ListedColormap(colorlist)
        norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)
       
        ax1.text(.5,1.02,mod1_name,horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax1.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhourm24+'-f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        if use_mod1 == 'a' and plot_nodata_text[0]:
          if fhr > 48 and dom not in ['hawaii', 'puerto_rico']:
              not_avail_text1a = "Not available\nat this forecast hour"
          else:
              not_avail_text1a = "Not Available"
          cs_1 = ax1.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=0.01,norm=norm)
          ax1.text(
              0.5, 0.5, not_avail_text1a, transform=ax1.transAxes, 
              fontsize=12, color='black',
              horizontalalignment='center', bbox=dict(
                  facecolor='white', alpha=0.8, 
                  boxstyle='round,pad=0.3'
              )
          )
        elif use_mod1 == 'b' and plot_nodata_text[1]:
          cs_1 = ax1.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=0.01,norm=norm)
          ax1.text(
              0.5, 0.5, 'Not Available', transform=ax1.transAxes, 
              fontsize=12, color='black',
              horizontalalignment='center', bbox=dict(
                  facecolor='white', alpha=0.8, 
                  boxstyle='round,pad=0.3'
              )
          )
        else:
            cs_1 = ax1.pcolormesh(use_lon1_shift,use_lat1_shift,use_qpf1,transform=transform,cmap=cm,vmin=0.01,norm=norm)
        cs_1.cmap.set_under('white',alpha=0.)
        cs_1.cmap.set_over('pink')
        cbar1 = fig.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar1.set_label(units,fontsize=6,labelpad=0)
        cbar1.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar1.ax.xaxis.set_tick_params(pad=0)
        cbar1.ax.tick_params(labelsize=6)
        ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

        ax2.text(.5,1.02,'NAM Nest',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax2.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhourm24+'-f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        if plot_nodata_text[2]:
          if fhr > 60:
              not_avail_text2 = "Not available\nat this forecast hour"
          else:
              not_avail_text2 = "Not Available"
          cs_2 = ax2.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=0.01,norm=norm)
          ax2.text(
              0.5, 0.5, not_avail_text2, transform=ax2.transAxes, 
              fontsize=12, color='black',
              horizontalalignment='center', bbox=dict(
                  facecolor='white', alpha=0.8, 
                  boxstyle='round,pad=0.3'
              )
          )
        else:
            cs_2 = ax2.pcolormesh(lon2_shift,lat2_shift,qpf_2,transform=transform,cmap=cm,vmin=0.01,norm=norm)
        cs_2.cmap.set_under('white',alpha=0.)
        cs_2.cmap.set_over('pink')
        cbar2 = fig.colorbar(cs_2,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar2.set_label(units,fontsize=6,labelpad=0)
        cbar2.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar2.ax.xaxis.set_tick_params(pad=0)
        cbar2.ax.tick_params(labelsize=6)
        ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

        ax3.text(.5,1.02,'RRFS',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax3.text(.5,0.95,itime+' '+cyc_model+'z cycle (f'+fhourm24+'-f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax3.text(.5,0.03,'Experimental Product - Not Official Guidance',horizontalalignment='center',fontsize=6,color='red',transform=ax3.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        if plot_nodata_text[3]:
          cs_3 = ax3.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=0.01,norm=norm)
          ax3.text(
              0.5, 0.5, 'Not Available', transform=ax3.transAxes, 
              fontsize=12, color='black',
              horizontalalignment='center', bbox=dict(
                  facecolor='white', alpha=0.8, 
                  boxstyle='round,pad=0.3'
              )
          )
        else:
            cs_3 = ax3.pcolormesh(lon3_shift,lat3_shift,qpf_3,transform=transform,cmap=cm,vmin=0.01,norm=norm)
        cs_3.cmap.set_under('white',alpha=0.)
        cs_3.cmap.set_over('pink')
        cbar3 = fig.colorbar(cs_3,ax=ax3,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar3.set_label(units,fontsize=6,labelpad=0)
        cbar3.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar3.ax.xaxis.set_tick_params(pad=0)
        cbar3.ax.tick_params(labelsize=6)
        ax3.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

        if dom in ['alaska', 'puerto_rico','hawaii']:
          ax4.text(.5,1.02,'MRMS',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        else:
          ax4.text(.5,1.02,'CCPA',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        ax4.text(.5,0.95,vtime_start+f' {cyc}z - '+vtime_end+f' {cyc}z',horizontalalignment='center',fontsize=6,transform=ax4.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
        if plot_nodata_text[5]:
          cs_4 = ax4.pcolormesh([[0]],[[0]],[[np.nan]],transform=transform,cmap=cm,vmin=0.01,norm=norm)
          ax4.text(
              0.5, 0.5, 'Not Available', transform=ax4.transAxes, 
              fontsize=12, color='black',
              horizontalalignment='center', bbox=dict(
                  facecolor='white', alpha=0.8, 
                  boxstyle='round,pad=0.3'
              )
          )
        else:
            cs_4 = ax4.pcolormesh(lon5,lat5,qpf_5,transform=transform,cmap=cm,vmin=0.01,norm=norm)
        cs_4.cmap.set_under('white',alpha=0.)
        cs_4.cmap.set_over('pink')
        cbar4 = fig.colorbar(cs_4,ax=ax4,orientation='horizontal',pad=0.01,shrink=0.85,ticks=[0.1,0.5,1,1.5,2,3,5,10,20],extend='max')
        cbar4.set_label(units,fontsize=6,labelpad=0)
        cbar4.ax.set_xticklabels([0.1,0.5,1,1.5,2,3,5,10,20])
        cbar4.ax.xaxis.set_tick_params(pad=0)
        cbar4.ax.tick_params(labelsize=6)
        ax4.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)


        convert_and_save(f'compareqpf{use_mod1}_'+dom+'_f'+fhour+'_'+pcpanl)
        t2 = time.perf_counter()
        t3 = round(t2-t1, 3)
        print(('%.3f seconds to plot 24-hr QPF with '+pcpanl+' for: '+dom+f'model{use_mod1}') % t3)

######################################################

main()
