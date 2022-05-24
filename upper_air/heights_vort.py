# ==========================================
# Title: Plot grib files for several models, Height contours and vorticity shading
# Author: Marcel Caron
# Date Modified: May 24 2022
# ==========================================

# Set Up

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cf
from datetime import datetime, timedelta as td
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
from metpy.units import units
import metpy.calc as mpcalc

# ============= CHANGES BELOW ==============

DATA_DIR = '/scratch2/NCEPDEV/ovp/Marcel.Caron/MEG/data/update'
SAVE_DIR = '/scratch2/NCEPDEV/stmp1/Marcel.Caron/cases/update/'

# ~~~~~~~
# This is the continuation of DATA_DIR, because each model directory is unique
# ~~~~~~~
gfsv15_string = 'gfs.%Y%m%d/%H/gfsv15/gfs.t%Hz.pgrb2.0p25.fFF' # put FF in place of lead time
#gfsv16_string = 'ovp/Marcel.Caron/MEG/data/MEG_20220113/gfs.%Y%m%d/%H/gfs.t%Hz.pgrb2.0p25.fFF' # put FF in place of lead time
gfsv16_string = 'gfs.%Y%m%d/%H/gfs.t%Hz.pgrb2.0p25.fFF' # put FF in place of lead time
gefs_string = 'ovp/Marcel.Caron/MEG/data/MEG_20220113/gefs.%Y%m%d/geavg.t%Hz.pgrb2a.0p50.fFF' # put FF in place of lead time
ec_string = 'stmp3/Alicia.Bentley/scripts/midwest/wgrbbul/ecmwf/U1D%m%d%H%M{VALIDM}{VALIDD}{VALIDH}{ANL}' # put FF in place of lead time, VALIDM, VALIDD, VALIDH, and ANL in place of valid month, day, and hour, and analysis indicator, resp.
nam_string = 'nam.%Y%m%d/nam.t%Hz.awphysFF.tm00.grib2' # put FF in place of lead time
nam_nest_string = 'nam.%Y%m%d/nam.t%Hz.conusnest.hiresfFF.tm00.grib2' # put FF in place of lead time
fv3_lam_string =  'ovp/Marcel.Caron/MEG/data/MEG_20210304/fv3lam.%Y%m%d/fv3lam.t%Hz.conus.fFF.grib2' # put FF in place of lead time
hiresw_arw_string = 'hiresw.%Y%m%d/hiresw.t%Hz.arw_5km.fFF.conus.grib2' # put FF in place of lead time
hiresw_arw2_string = 'ovp/Marcel.Caron/MEG/data/MEG_20210304/hiresw.%Y%m%d/hiresw.t%Hz.arw_5km.fFF.conusmem2.grib2' # put FF in place of lead time
hiresw_nmmb_string = 'ovp/Marcel.Caron/MEG/data/MEG_20210304/hiresw.%Y%m%d/hiresw.t%Hz.nmmb_5km.fFF.conus.grib2' # put FF in place of lead time
href_mean_string = 'ovp/Marcel.Caron/MEG/data/MEG_20210304/href.%Y%m%d/href.t%Hz.conus.mean.fFF.grib2' # put FF in place of lead time
href_pmmn_string = 'ovp/Marcel.Caron/MEG/data/MEG_20210304/href.%Y%m%d/href.t%Hz.conus.pmmn.fFF.grib2' # put FF in place of lead time
href_avrg_string = 'ovp/Marcel.Caron/MEG/data/MEG_20210304/href.%Y%m%d/href.t%Hz.conus.avrg.fFF.grib2' # put FF in place of lead time
rap_string = 'ovp/Marcel.Caron/MEG/data/MEG_20210304/rap.%Y%m%d/rap.t%Hz.awp130pgrbfFF.grib2' # put FF in place of lead time
hrrr_string = 'hrrr.%Y%m%d/hrrr.t%Hz.wrfprsfFF.grib2' # put FF in place of lead time

# ~~~~~~~
# valid time
# ~~~~~~~
valid = datetime(2022,2,16,0,0,0) 

# ~~~~~~~
# Create a separate plot for each of the following models
# e.g., ['st4', 'nam', 'nam_nest', 'fv3_lam', 'hiresw_arw', 'hiresw_arw2', 'hiresw_nmmb', 'href', 'rap', 'hrrr', 'gfsv15', 'gfsv16', 'gefs', 'ec']
# ~~~~~~~
models = ['gfsv16'] 

# ~~~~~~~
# Create a separate plot for each of the following init times
# ~~~~~~~
inits = [datetime(2022,2,14,0,0,0),
         datetime(2022,2,16,0,0,0)]

# ~~~~~~~
# Create a separate plot for each of the following domains
# ~~~~~~~
domains = ['global'] # northamer, conus, econus, zoom, neast, nwheat, splains_zoom, other

# ~~~~~~~
# if domain = 'other', will use the parameters below to plot, otherwise ignores
# ~~~~~~~
latrange = (-90., 90.)
lonrange = (-180., 180.)
parallels = (32., 46.)
merid = -105.
figsize = (9.3,7.) # tuple width, height for desired dimensions of plots

# ============= CHANGES ABOVE ==============

# Functions

# input types are datetime, datetime, timedelta objects
def daterange(start, end, td):
   curr = start
   while curr <= end:
      yield curr
      curr+=td

# returns data value interpolated at lat and lon, using lats and lons meshgrids
def bilinear_interp(data, lats, lons, lat, lon):
   points = (lats.flatten(), lons.flatten())
   values = data.flatten()
   xi = (lat, lon)
   method = 'linear'
   return griddata(points, values, xi, method=method)

# Works generally if you're plotting in the western hemisphere
def get_lons(da):
   if np.max(da.longitude) > 180.:
      lons = np.subtract(np.mod(np.add(np.array(da.longitude),180.),360.),180.)
   else:
      lons = np.array(da.longitude)
   return lons


def get_domains(domain='northamer', parallels=None, merid=None, latrange=None, lonrange=None, num=0, figsize=None, return_box=False):
   if str(domain).upper() == 'NORTHAMER':
      if not return_box:
         fig = plt.figure(num=num, figsize=(7.6,7.))
      parallels = (23., 37.)
      merid = -95.
      latrange = (8., 72.)
      lonrange = (-151., -49.)
   elif str(domain).upper() == 'GLOBAL':
      if not return_box:
         fig = plt.figure(num=num, figsize=(9.3,7.))
      parallels = (-10., 10.)
      merid = -105.
      latrange = (-90., 90.)
      lonrange = (-180., 180.)
   elif str(domain).upper() == 'CONUS':
      if not return_box:
         fig = plt.figure(num=num, figsize=(9.1,7.))
      parallels = (25., 35.)
      merid = -97.5
      latrange = (19., 57.)
      lonrange = (-123., -70.)
   elif str(domain).upper() == 'NWHEAT':
      if not return_box:
         fig = plt.figure(num=num, figsize=(9.3,7.))
      parallels = (32., 46.)
      merid = -105.
      latrange = (17.5, 64.)
      lonrange = (-145., -77.5)
   elif str(domain).upper() == 'NEAST':
      if not return_box:
         fig = plt.figure(num=num, figsize=(7.6,7.))
      parallels = (25., 35.)
      merid = -77.
      latrange = (35.1, 47.1)
      lonrange = (-85.5, -69.)
   elif str(domain).upper() == 'SPLAINS_ZOOM':
      if not return_box:
         fig = plt.figure(num=num, figsize=(9.5,7.))
      parallels = (25., 35.)
      merid = -98.
      latrange = (30.5, 40.)
      lonrange = (-104.5, -89.)
   elif str(domain).upper() == 'ZOOM':
      if not return_box:
         fig = plt.figure(num=num, figsize=(7.6,7.))
      parallels = (25., 35.)
      merid = -90.
      latrange = (25., 50.)
      lonrange = (-108., -72.)
   elif str(domain).upper() == 'ECONUS':
      if not return_box:
         fig = plt.figure(num=num, figsize=(9.5,7.))
      parallels = (25., 35.)
      merid = -90.
      latrange = (23.6, 47.4)
      lonrange = (-105., -62.)
   elif str(domain).upper() == 'OTHER':
      if not return_box:
         fig = plt.figure(num=num, figsize=figsize)
      parallels = parallels
      merid = merid
      latrange = latrange
      lonrange = lonrange
   #proj = ccrs.LambertConformal(central_longitude=merid, standard_parallels=parallels)
   proj = ccrs.PlateCarree(central_longitude=merid)
   '''
   m = Basemap(llcrnrlon=lonrange[0], urcrnrlon=lonrange[1], llcrnrlat=latrange[0], urcrnrlat=latrange[1], \
               resolution='i', projection='lcc', lat_1=parallels[0], lat_2=parallels[1], lon_0=merid, \
               area_thresh=100.)
   '''
   if return_box:
      return lonrange, latrange
   else:
      return proj, fig

def gfs_upper_data(init, model, model_string, fvalid):
   
   print('========================================')
   print(f'Getting {model} Heights and Vorticity Data')
   #Get and sum absv and gph data
   if str(model).upper() in ['GFSV15','GFSV16','GEFS','EC']: # global models have 3 leading zeros *smh*
      fpath = os.path.join(DATA_DIR, init.strftime(model_string).replace('FF','{0:0=3d}'.format(fvalid))) 
      if str(model).upper() == 'EC':
         valid = init + td(hours=fvalid)
         fpath = fpath.replace('{VALIDY}', valid.strftime('%Y')).replace('{VALIDM}', valid.strftime('%m')).replace('{VALIDD}', valid.strftime('%d')).replace('{VALIDH}', valid.strftime('%H'))
         if int(fvalid) == 0:
            fpath = fpath.replace('{ANL}','011')
         else:
            fpath = fpath.replace('{ANL}','001')
         index_500_absv = 5
         index_500_gh = 7
      elif str(model).upper() == 'GEFS':
         index_500_absv = 4
         index_500_gh = index_500_absv
      else:
         index_500_absv = 12
         index_250_absv = 17
         index_500_gh = index_500_absv
         index_250_gh = index_250_absv
   else:
      fpath = os.path.join(DATA_DIR, init.strftime(model_string).replace('FF','{0:0=2d}'.format(fvalid)))
      if str(model).upper() == 'NAM':
         index_500_absv = 3
         index_500_gh = 20
      elif str(model).upper() == 'NAM_NEST':
         index_500_absv = 4
         index_250_absv = 6
         index_500_gh = 20
         index_250_gh = 30
      elif str(model).upper() == 'HRRR':
         index_500_absv = 21
         index_250_absv = 31
         index_500_gh = index_500_absv
         index_250_gh = index_250_absv
      else:
         raise OSError('Requested a model that isn\'t supported by this script. Skipping ...')
   if str(model).upper() == 'EC':
      with xr.open_dataset(fpath, engine='cfgrib', backend_kwargs=dict(
         indexpath='',
         filter_by_keys={'edition':1,'typeOfLevel':'isobaricInhPa','shortName':'u'})) as ds_u:
         u_500 = ds_u.u[index_500_absv]
         lons = get_lons(u_500)
         lats = np.array(u_500.latitude)
         data_u = np.array(u_500)
      with xr.open_dataset(fpath, engine='cfgrib', backend_kwargs=dict(
         indexpath='',
         filter_by_keys={'edition':1,'typeOfLevel':'isobaricInhPa','shortName':'v'})) as ds_v:
         v_500 = ds_v.v[index_500_absv]
         data_v = np.array(v_500)
      with xr.open_dataset(fpath, engine='cfgrib', backend_kwargs=dict(
         indexpath='',
         filter_by_keys={'edition':1,'typeOfLevel':'isobaricInhPa','shortName':'gh'})) as ds_gh:
         try:
            gh = ds_gh.gh[index_500_gh]
            if gh.isobaricInhPa != 500.:
               raise ValueError
         except ValueError:
            gh = ds_gh.gh[index_500_gh-2]
         data_gh = np.array(gh)
      mlons, mlats = np.meshgrid(lons, lats)
      dx, dy = mpcalc.lat_lon_grid_deltas(mlons, mlats)
      data_absv = mpcalc.absolute_vorticity(u_500, v_500, dx, dy, mlats*units.degrees, dim_order='yx')
   else:
      if str(model).upper() == 'GEFS':
         with xr.open_dataset(fpath, engine='cfgrib', backend_kwargs=dict(
            indexpath='',
            filter_by_keys={'typeOfLevel':'isobaricInhPa','shortName':'u'})) as ds_u:
            u_500 = ds_u.u[index_500_absv]
            lons = get_lons(u_500)
            lats = np.array(u_500.latitude)
            data_u = np.array(u_500)
         with xr.open_dataset(fpath, engine='cfgrib', backend_kwargs=dict(
            indexpath='',
            filter_by_keys={'typeOfLevel':'isobaricInhPa','shortName':'v'})) as ds_v:
            v_500 = ds_v.v[index_500_absv]
            data_v = np.array(v_500)
         mlons, mlats = np.meshgrid(lons, lats)
         dx, dy = mpcalc.lat_lon_grid_deltas(mlons, mlats)
         data_absv = mpcalc.absolute_vorticity(u_500, v_500, dx, dy, mlats*units.degrees, dim_order='yx')
      else:
         with xr.open_dataset(fpath, engine='cfgrib', backend_kwargs=dict(
            indexpath='',
            filter_by_keys={'typeOfLevel':'isobaricInhPa','shortName':'absv'})) as ds_absv:
            absv = ds_absv.absv[index_250_absv] # absolute vorticity, index to the pressure level (e.g., 12 for 500 hPa)
            lons = get_lons(absv)
            lats = np.array(absv.latitude)
            data_absv = np.array(absv)
      with xr.open_dataset(fpath, engine='cfgrib', backend_kwargs=dict(
         indexpath='',
         filter_by_keys={'typeOfLevel':'isobaricInhPa','shortName':'gh'})) as ds_gh:
         gh = ds_gh.gh[index_250_gh] # geopotential height, index to the pressure level (e.g., 12 for 500 hPa)
         data_gh = np.array(gh)
   
   data_absv*=1E5 # convert s^-1 to 10^-5 s^-1
   data_gh/=10. # convert m to dam   
   
   print('Data Retrieved ' + u'\u2713')
   print('========================================') 
   return (data_absv, data_gh, lats, lons)

# Returns numpy data meshgrid as a bilinear interpolation of model data onto analysis grid
def diff_data(anl_lons, anl_lats, model_lons, model_lats, model_data):
   anl_lons, anl_lats, model_lons, model_lats, model_data = [np.array(item) for item in [anl_lons, anl_lats, model_lons, model_lats, model_data]]
   model_data_interpd = griddata(
      (model_lons.flatten(), model_lats.flatten()),
      model_data.flatten(),
      (anl_lons, anl_lats), method='linear')
   return model_data_interpd

def plot_vort_height(num, model, lons, lats, data_hgt, data_absv, valid, parallels=None, merid=None, latrange=None, lonrange=None, figsize=None, init=None):
   print(f'Plot {num} ... 250 hPa Height and Absolute Vorticity plot for {str(model).upper()}')
   print('========================================')
   print('Prepping Basemap')
   # Get the basemap object
   if str(domain).upper() == 'OTHER':
      #global parallels, merid, latrange, lonrange, figsize
      proj, fig = get_domains(domain=domain, parallels=parallels, merid=merid, latrange=latrange, lonrange=lonrange, figsize=figsize, num=num)
   else:
      proj, fig = get_domains(domain=domain, num=num)
      lonrange, latrange = get_domains(domain=domain, return_box=True)
   
   ax = fig.add_axes([.05,.05,.9,.9], projection=proj)
   ax.set_extent([lonrange[0], lonrange[1], latrange[0], latrange[1]], crs=ccrs.PlateCarree())
   countries = cf.NaturalEarthFeature(
      category='cultural',
      name='admin_0_boundary_lines_land',
      scale='50m',
      facecolor='none')
   states=cf.NaturalEarthFeature(
      category='cultural',
      name='admin_1_states_provinces_lines',
      scale='50m',
      facecolor='none')
   coastlines = cf.NaturalEarthFeature(
      category='physical',
      name='coastline',
      scale='50m',
      facecolor='none')
   ocean = cf.NaturalEarthFeature(
      category='physical',
      name='ocean',
      scale='50m',
      facecolor='none')
   land = cf.NaturalEarthFeature(
      category='physical',
      name='land',
      scale='50m',
      facecolor='none')
   lakes = cf.NaturalEarthFeature(
      category='physical',
      name='lakes',
      scale='50m',
      facecolor='none')
   ax.add_feature(land, facecolor='#d3d3d3', zorder=2)
   ax.add_feature(ocean, zorder=2, edgecolor='black', linewidth=.7, facecolor='#5c5c5c')
   ax.add_feature(lakes, edgecolor='black', facecolor='#5c5c5c', linewidth=.5, zorder=2)
   ax.add_feature(countries, edgecolor='black', facecolor='none', linewidth=.7, zorder=4)
   ax.add_feature(states, edgecolor='black', facecolor='none', linewidth=.5, zorder=6)
   ax.add_feature(coastlines, edgecolor='black', facecolor='none', linewidth=.4, zorder=5)
   latlongrid=5.
   parallels = list(np.arange(0., 90., latlongrid))
   meridians = list(np.arange(180., 360., latlongrid))

   '''
   # Prep the basemap
   m.drawcoastlines(zorder=5)
   m.drawmapboundary(fill_color='#5c5c5c', zorder=1) # fill everything within domain
   m.fillcontinents(color='#D3D3D3', lake_color='#5c5c5c', zorder=2) # fill continents and lakes
   #m.drawcounties() # does not work!!!
   m.drawstates(linewidth=0.4, zorder=6)
   m.drawcountries(zorder=4)
   latlongrid=5.
   parallels = np.arange(0., 90., latlongrid)
   meridians = np.arange(180., 360., latlongrid)
   m.drawparallels(parallels, labels=[1,0,0,0], linewidth=.4, fontsize=9, zorder=7)
   m.drawmeridians(meridians, labels=[0,0,0,1], linewidth=.4, fontsize=9, zorder=8)
   '''   

   print('Basemap Prepped ' + u'\u2713')
   print('Plotting Vorticity ...')
   clevs = [-999., 16., 20., 24., 28., 32., 36., 40., 44., 48., 999.]
   MEG_vort_colors = [
      '#ffffff', # < 16. 10^-5 seconds^-1
      '#fbd401', # 16. - 20. 10^-5 seconds^-1 
      '#fdba02', # 20. - 24. 10^-5 seconds^-1 
      '#ffa305', # 24. - 28. 10^-5 seconds^-1 
      '#ff6b0c', # 28. - 32. 10^-5 seconds^-1 
      '#ff3413', # 32. - 36. 10^-5 seconds^-1 
      '#f8263b', # 36. - 40. 10^-5 seconds^-1 
      '#f02764', # 40. - 44. 10^-5 seconds^-1 
      '#f731aa', # 44. - 48. 10^-5 seconds^-1 
      '#ff61ef', # > 48. 10^-5 seconds^-1 
   ]
   cmap = colors.ListedColormap(MEG_vort_colors)
   cmap.set_under(color='white',alpha=0.)
   norm = colors.BoundaryNorm(clevs, cmap.N, clip=False)

   # Mask height data that are out of bounds; helps clabel better-populate the domain with contour labels
   contour_ints = np.arange(850,11000,6)
   ax_ul = (0.,1.)
   ax_ur = (1.,1.)
   ax_ul_display = ax.transAxes.transform(ax_ul)
   ax_ur_display = ax.transAxes.transform(ax_ur)
   ax_ul_data = ax.transData.inverted().transform(ax_ul_display)
   ax_ur_data = ax.transData.inverted().transform(ax_ur_display)
   ax_ul_cartesian = ccrs.PlateCarree().transform_point(*ax_ul_data, src_crs=proj)
   ax_ur_cartesian = ccrs.PlateCarree().transform_point(*ax_ur_data, src_crs=proj)
   mask1 = lons < ax_ul_cartesian[0] # masking everything outside North Am. (may need to refine this someday)
   mask2 = lons > ax_ur_cartesian[0]
   mask3 = lats > latrange[1]+5.
   mask4 = lats < latrange[0]-5.
   mask = mask1+mask2+mask3+mask4 
   mdata_hgt = np.ma.MaskedArray(data_hgt,mask=mask)
   mdata_absv = np.ma.MaskedArray(data_absv, mask=mask)

   if str(domain).upper() == "GLOBAL":
      plot_vort = ax.contourf(lons, lats, data_absv, shading='flat', levels=clevs, cmap=cmap, vmin=16., vmax=48., zorder=3, transform=ccrs.PlateCarree(), norm=norm) #clev=clevs]
      #plot_vort = ax.pcolormesh(lons, lats, data_absv, shading='flat', cmap=cmap, vmin=16., vmax=48., zorder=3, transform=ccrs.PlateCarree(), norm=norm) #clev=clevs]
      print('Plotting Heights ...')
      plot_hgt = ax.contour(lons, lats, data_hgt, levels=contour_ints, colors='black', linewidths=1.5, transform=ccrs.PlateCarree(), zorder=9)
   else:
      plot_vort = ax.contourf(lons, lats, mdata_absv, shading='flat', levels=clevs, cmap=cmap, vmin=16., vmax=48., zorder=3, transform=ccrs.PlateCarree(), norm=norm) #clev=clevs]
      #plot_vort = ax.pcolormesh(lons, lats, mdata_absv, shading='flat', cmap=cmap, vmin=16., vmax=48., zorder=3, transform=ccrs.PlateCarree(), norm=norm) #clev=clevs]
      print('Plotting Heights ...')
      plot_hgt = ax.contour(lons, lats, mdata_hgt, levels=contour_ints, colors='black', linewidths=1.5, transform=ccrs.PlateCarree(), zorder=9)
   plt.clabel(plot_hgt, contour_ints, fmt='%i', fontsize=8.5, inline=1, inline_spacing=0,  manual=False)

   print('Plotted ' + u'\u2713')
   model_name = model.upper().replace('_', ' ')
   title_string_left = f'{model_name} | ' + init.strftime('initialized %HZ %d %B %Y') + valid.strftime(' valid %HZ %d %B %Y')
   title_string_right = r'250-hPa g (dam), Vorticity ($10^{-5} s^{-1}$)'
   plt.title(title_string_left, loc='left', fontsize=7)
   plt.title(title_string_right, loc='right', fontsize=7)

   fig.subplots_adjust(left=.05, right=.93, top=.95, bottom=.05, wspace=0.05, hspace=0)
   cax = fig.add_axes([.93, .05, .01, .9])
   cbar_ticks = [16., 20., 24., 28., 32., 36., 40., 44., 48.]
   cb = plt.colorbar(plot_vort, orientation='vertical', cax=cax, cmap=cmap, norm=norm, boundaries=clevs, spacing='uniform', ticks=cbar_ticks, drawedges=True)
   cb.dividers.set_color('black')
   cb.dividers.set_linewidth(2)
   cb.ax.tick_params(labelsize=8, labelright=True, labelleft=False, right=False)
   cb.ax.set_yticklabels(['16.0','20.0','24.0','28.0','32.0','36.0','40.0','44.0','48.0',''])
   cax.hlines([0, 1], 0, 1, colors='black', linewidth=4)

   save_string = f'{model}_Z250_VORT_{domain}.'+init.strftime('init_%Y%m%d%H.')+valid.strftime('valid_%Y%m%d%H.png')
   save_path = os.path.join(SAVE_DIR, save_string)
   plt.savefig(save_path, bbox_inches='tight')
   print(f'Plot {num} saved successfully in {save_path}')
   plt.close(num)
   print('========================================')

# Main
def main():
   # Get, Process, and Plot Data
   # Each dataset needs to be processed separately because they contain different precip accumulation step ranges
   num = 0
   global domain   
   for domain in domains:
      if str(domain).upper() != 'OTHER':
         lonrange, latrange = get_domains(domain=domain, return_box=True)
      for model in models:
         for init in inits:
            try: # if a file isn't found, we'll ignore this init and go to the next one
               fvalid = int((valid-init).days*24 + (valid-init).seconds/3600) # lead time in hours
               if str(model).upper() == 'NAM':
                  nam_absv, nam_hgt, nam_lats, nam_lons = gfs_upper_data(init, str(model).upper(), nam_string, fvalid)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, nam_lons[1:-1,1:-1], nam_lats[1:-1,1:-1], nam_hgt[1:-1,1:-1], nam_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, nam_lons[1:-1,1:-1], nam_lats[1:-1,1:-1], nam_hgt[1:-1,1:-1], nam_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'NAM_NEST':
                  nam_nest_absv, nam_nest_hgt, nam_nest_lats, nam_nest_lons = gfs_upper_data(init, str(model).upper(), nam_nest_string, fvalid)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, nam_nest_lons[1:-1,1:-1], nam_nest_lats[1:-1,1:-1], nam_nest_hgt[1:-1,1:-1], nam_nest_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, nam_nest_lons[1:-1,1:-1], nam_nest_lats[1:-1,1:-1], nam_nest_hgt[1:-1,1:-1], nam_nest_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'FV3_LAM':
                  fv3_lam_absv, fv3_lam_hgt, fv3_lam_lats, fv3_lam_lons = gfs_upper_data(init, str(model).upper(), fv3_lam_string, fvalid)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, fv3_lam_lons[1:-1,1:-1], fv3_lam_lats[1:-1,1:-1], fv3_lam_hgt[1:-1,1:-1], fv3_lam_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, fv3_lam_lons[1:-1,1:-1], fv3_lam_lats[1:-1,1:-1], fv3_lam_hgt[1:-1,1:-1], fv3_lam_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'HIRESW_ARW':
                  hiresw_arw_absv, hiresw_arw_hgt, hiresw_arw_lats, hiresw_arw_lons = gfs_upper_data(init, str(model).upper(), hiresw_arw_string, fvalid)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, hiresw_arw_lons[1:-1,1:-1], hiresw_arw_lats[1:-1,1:-1], hiresw_arw_hgt[1:-1,1:-1], hiresw_arw_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, hiresw_arw_lons[1:-1,1:-1], hiresw_arw_lats[1:-1,1:-1], hiresw_arw_hgt[1:-1,1:-1], hiresw_arw_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'HIRESW_ARW2':
                  hiresw_arw2_absv, hiresw_arw2_hgt, hiresw_arw2_lats, hiresw_arw2_lons = gfs_upper_data(init, str(model).upper(), hiresw_arw2_string, fvalid)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, hiresw_arw2_lons[1:-1,1:-1], hiresw_arw2_lats[1:-1,1:-1], hiresw_arw2_hgt[1:-1,1:-1], hiresw_arw2_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, hiresw_arw2_lons[1:-1,1:-1], hiresw_arw2_lats[1:-1,1:-1], hiresw_arw2_hgt[1:-1,1:-1], hiresw_arw2_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'HIRESW_NMMB':
                  hiresw_nmmb_absv, hiresw_nmmb_hgt, hiresw_nmmb_lats, hiresw_nmmb_lons = gfs_upper_data(init, str(model).upper(), hiresw_nmmb_string, fvalid)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, hiresw_nmmb_lons[1:-1,1:-1], hiresw_nmmb_lats[1:-1,1:-1], hiresw_nmmb_hgt[1:-1,1:-1], hiresw_nmmb_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, hiresw_nmmb_lons[1:-1,1:-1], hiresw_nmmb_lats[1:-1,1:-1], hiresw_nmmb_hgt[1:-1,1:-1], hiresw_nmmb_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'HREF':
                  try:
                     href_mean_data, href_mean_lats, href_mean_lons = accum_hourly_precip_data(init, str('HREF_MEAN').upper(), href_mean_string, fstart, fend, 3)
                     plot_precip(num, 'href_mean', href_mean_lons, href_mean_lats, href_mean_data, init=init)
                     num+=1
                     if diff_plots and 'st4'.upper() in [str(model).upper() for model in models]:
                        href_mean_interp = diff_data(st4_lons, st4_lats, href_mean_lons, href_mean_lats, href_mean_data)
                        href_mean_diff = np.subtract(href_mean_interp, st4_data)
                        plot_diff(num, 'href_mean', st4_lons, st4_lats, href_mean_diff, init=init)
                        num+=1
                  except OSError as e:
                     continue
                  try:
                     href_pmmn_data, href_pmmn_lats, href_pmmn_lons = accum_hourly_precip_data(init, str('HREF_PMMN').upper(), href_pmmn_string, fstart, fend, 3)
                     plot_precip(num, 'href_pmmn', href_pmmn_lons, href_pmmn_lats, href_pmmn_data, init=init)
                     num+=1
                     if diff_plots and 'st4'.upper() in [str(model).upper() for model in models]:
                        href_pmmn_interp = diff_data(st4_lons, st4_lats, href_pmmn_lons, href_pmmn_lats, href_pmmn_data)
                        href_pmmn_diff = np.subtract(href_pmmn_interp, st4_data)
                        plot_diff(num, 'href_pmmn', st4_lons, st4_lats, href_pmmn_diff, init=init)
                        num+=1
                  except OSError as e:
                     continue
                  try:
                     href_avrg_data, href_avrg_lats, href_avrg_lons = accum_hourly_precip_data(init, str('HREF_AVRG').upper(), href_avrg_string, fstart, fend, 3)
                     plot_precip(num, r'href_avrg', href_avrg_lons, href_avrg_lats, href_avrg_data, init=init)
                     num+=1
                     if diff_plots and 'st4'.upper() in [str(model).upper() for model in models]:
                        href_avrg_interp = diff_data(st4_lons, st4_lats, href_avrg_lons, href_avrg_lats, href_avrg_data)
                        href_avrg_diff = np.subtract(href_avrg_interp, st4_data)
                        plot_diff(num, r'href_avrg', st4_lons, st4_lats, href_avrg_diff, init=init)
                        num+=1
                  except OSError as e:
                     continue
               if str(model).upper() == 'RAP':
                  rap_absv, rap_hgt, rap_lats, rap_lons = gfs_upper_data(init, str(model).upper(), rap_string, fvalid)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, rap_lons[1:-1,1:-1], rap_lats[1:-1,1:-1], rap_hgt[1:-1,1:-1], rap_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, rap_lons[1:-1,1:-1], rap_lats[1:-1,1:-1], rap_hgt[1:-1,1:-1], rap_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'HRRR':
                  hrrr_absv, hrrr_hgt, hrrr_lats, hrrr_lons = gfs_upper_data(init, str(model).upper(), hrrr_string, fvalid)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, hrrr_lons[1:-1,1:-1], hrrr_lats[1:-1,1:-1], hrrr_hgt[1:-1,1:-1], hrrr_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, hrrr_lons[1:-1,1:-1], hrrr_lats[1:-1,1:-1], hrrr_hgt[1:-1,1:-1], hrrr_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'GFSV15':
                  gfsv15_absv, gfsv15_hgt, gfsv15_y, gfsv15_x = gfs_upper_data(init, str(model).upper(), gfsv15_string, fvalid)
                  gfsv15_lons, gfsv15_lats = np.meshgrid(gfsv15_x, gfsv15_y)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, gfsv15_lons[1:-1,1:-1], gfsv15_lats[1:-1,1:-1], gfsv15_hgt[1:-1,1:-1], gfsv15_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, gfsv15_lons[1:-1,1:-1], gfsv15_lats[1:-1,1:-1], gfsv15_hgt[1:-1,1:-1], gfsv15_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'GFSV16':
                  gfsv16_absv, gfsv16_hgt, gfsv16_y, gfsv16_x = gfs_upper_data(init, str(model).upper(), gfsv16_string, fvalid)
                  gfsv16_lons, gfsv16_lats = np.meshgrid(gfsv16_x, gfsv16_y)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, gfsv16_lons[1:-1,1:-1], gfsv16_lats[1:-1,1:-1], gfsv16_hgt[1:-1,1:-1], gfsv16_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, gfsv16_lons[1:-1,1:-1], gfsv16_lats[1:-1,1:-1], gfsv16_hgt[1:-1,1:-1], gfsv16_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'EC':
                  ec_absv, ec_hgt, ec_y, ec_x = gfs_upper_data(init, str(model).upper(), ec_string, fvalid)
                  ec_lons, ec_lats = np.meshgrid(ec_x, ec_y)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, ec_lons[1:-1,1:-1], ec_lats[1:-1,1:-1], ec_hgt[1:-1,1:-1], ec_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, ec_lons[1:-1,1:-1], ec_lats[1:-1,1:-1], ec_hgt[1:-1,1:-1], ec_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
               if str(model).upper() == 'GEFS':
                  gefs_absv, gefs_hgt, gefs_y, gefs_x = gfs_upper_data(init, str(model).upper(), gefs_string, fvalid)
                  gefs_lons, gefs_lats = np.meshgrid(gefs_x, gefs_y)
                  # Basemap doesn't handle edges of these global domains well, so we're tossing out the edges when plotting
                  if str(domain).upper() == 'OTHER':
                     plot_vort_height(num, model, gefs_lons[1:-1,1:-1], gefs_lats[1:-1,1:-1], gefs_hgt[1:-1,1:-1], gefs_absv[1:-1,1:-1], valid, domain=domain, lonrange=lonrange, latrange=latrange, parallels=parallels, merid=merid, init=init) 
                  else:
                     plot_vort_height(num, model, gefs_lons[1:-1,1:-1], gefs_lats[1:-1,1:-1], gefs_hgt[1:-1,1:-1], gefs_absv[1:-1,1:-1], valid, init=init) 
                  num+=1
            except OSError as e:
               print(e)
               print("Continuing ...")
               continue

main()
