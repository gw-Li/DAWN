import numpy              as     np
import xarray             as     xr
from   icecream           import ic
import matplotlib.pyplot  as     plt
from   matplotlib.patches import Rectangle

# Please modify the later parameter:
i, j           = 55,55         # Which grid you want to plot?
vname          = 'T2MAX'       # The variable you want to show
init_year      = 2024
init_month     = 9
init_day       = 3

def draw_colored_box(ax, x_start, x_end, y_start, y_end, pdf_values, pec33=None, pec66=None, border_color='k'):
    """
    Draws a box on the given Axes (ax) from (x_start) to (x_end) horizontally
    and (y_start) to (y_end) vertically, subdivided into horizontal strips
    colored by pdf_values using the 'gray_r' colormap.
    Optionally, draws horizontal lines at the positions of pec33 and pec66
    (if provided and if they lie within [y_start, y_end]) using the same color as border_color.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the colored box.
    x_start, x_end : float
        Horizontal start and end coordinates of the box.
    y_start, y_end : float
        Vertical start and end coordinates of the box.
    pdf_values : 1D array-like
        Values used for coloring each horizontal subdivision.
    pec33 : float, optional
        The value at which to draw the first horizontal line (default is None).
    pec66 : float, optional
        The value at which to draw the second horizontal line (default is None).
    border_color : str, optional
        Color name or code for the border of the box and the extra lines (default is 'k' = black).
    """
    # Prepare the colormap and normalization
    cmap = plt.cm.get_cmap('gray_r')
    norm = plt.Normalize(vmin=np.min(pdf_values), vmax=np.max(pdf_values))
    
    # Calculate dimensions
    width  = x_end - x_start
    height = y_end - y_start
    
    # Number of slices
    n_slices = len(pdf_values)
    # Height of each slice
    dy = height / n_slices
    
    # Draw each horizontal strip
    for idx, val in enumerate(pdf_values):
        y_bottom = y_start + idx * dy
        color_val = cmap(norm(val))
        
        rect = Rectangle(
            (x_start, y_bottom),  # bottom-left corner
            width,                # rectangle width
            dy,                   # rectangle height
            facecolor=color_val,
            edgecolor='none'
        )
        ax.add_patch(rect)
    
    # Draw the outline border of the box
    border_rect = Rectangle(
        (x_start, y_start),
        width,
        height,
        fill=False,               # no fill
        edgecolor=border_color,   # use the specified border color
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(border_rect)
    
    # Optionally draw horizontal lines at the locations of pec33 and pec66
    for value in [pec33, pec66]:
        if value is not None and y_start <= value <= y_end:
            ax.plot(
                [x_start, x_end], 
                [value, value],
                color=border_color,
                linewidth=1.5,
                zorder=3
            )

# Read the dataset
p_obs          = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/Data_Viewer/'
p_cwrf         = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/'
path_operational = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/'
path_viewer    = '/mnt/gfs01/PUB/S2S/V2023-07/Operational/Data_Viewer/'

months_str     = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
# Read the observation and CWRF dataset
ds_obs         = xr.open_dataset(f'{p_obs}{vname}_OBS_PDF.nc')
ds_cwrf        = xr.open_dataset(f'{p_cwrf}{init_year}{init_month:02}{init_day:02}/{init_year}{init_month:02}{init_day:02}_{vname}_PDF.nc', engine='netcdf4')
tgt_months     = ds_cwrf['time'].dt.month.values
# Read the 33 and 66 percentage
ds_cwrf_v1 = xr.open_dataset(f'{path_operational}{init_year}{init_month:02}{init_day:02}/{init_year}{init_month:02}{init_day:02}_for_data_viewer.nc')
da_cwrf_33 = ds_cwrf_v1['percentage_less_33'].sel(vname=vname)[:,i,j]
da_cwrf_66 = ds_cwrf_v1['percentage_abov_66'].sel(vname=vname)[:,i,j]
ds_obs_v1  = xr.open_dataset(f'{path_viewer}/{vname}_OBS_quantile.nc')
da_obs_33  = ds_obs_v1['da_quantile_33'][:,i,j]
da_obs_66  = ds_obs_v1['da_quantile_66'][:,i,j]

list_obs_grid, list_obs_pdf, list_cwrf_grid, list_cwrf_pdf = [],[],[],[]
lsit_obs33, list_obs66 ,lsit_cwrf33, list_cwrf66           = [],[],[],[]
for imonth in range(6):
    # Select the target month
    tgt_month      = tgt_months[imonth]
    ds_obs_month   = ds_obs.sel(month = tgt_month)
    ds_cwrf_month  = ds_cwrf.sel(time=ds_cwrf['time'].dt.month == tgt_month)
    # OBS PDF in given grid point
    list_obs_grid.append(ds_obs_month['x_grid'][i,j,:].values)
    list_obs_pdf.append( ds_obs_month['pdf'][i,j,:].values)
    # CWRF PDF
    list_cwrf_grid.append(ds_cwrf_month['x_grid'][0,0,i,j,:].values)
    list_cwrf_pdf.append( ds_cwrf_month['pdf'][0,0,i,j,:].values)
    lsit_obs33.append( da_obs_33.sel( month=tgt_month).values.item()),list_obs66.append( da_obs_66.sel( month=tgt_month).values.item())
    lsit_cwrf33.append(da_cwrf_33.sel(month=tgt_month).values.item()),list_cwrf66.append(da_cwrf_66.sel(month=tgt_month).values.item())

# Calculate the range with some space
max_value =max( max(max(row) for row in list_obs_grid),max(max(row) for row in list_cwrf_grid) )
min_value =min( min(min(row) for row in list_obs_grid),min(min(row) for row in list_cwrf_grid) )
y_min = (int(min_value) // 10) * 10
y_max = (int(max_value) // 10 + 1) * 10


# Create the plot
fig, ax = plt.subplots(figsize=(10,6.18))
ax.set_xlim([0, 96])
ax.set_ylim([y_min, y_max])

# Example: custom x-axis ticks and labels
x_positions = [8, 24, 40,58 ,72,88]
x_labels    = [months_str[kk-1] for kk in tgt_months]
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels)

for imonth in range(6):
    # Call the function for Observation with a green border
    draw_colored_box(
        ax, 
        x_start=imonth*16+3, x_end=imonth*16+8, 
        y_start=list_obs_grid[imonth].min(), y_end=list_obs_grid[imonth].max(), 
        pdf_values=list_obs_pdf[imonth], 
        pec33  =lsit_obs33[imonth], pec66=list_obs66[imonth],
        border_color='steelblue'
    )

    # Call the function again for CWRF with a red border
    draw_colored_box(
        ax, 
        x_start=imonth*16+8, x_end=imonth*16+13,
        y_start=list_cwrf_grid[imonth].min(), y_end=list_cwrf_grid[imonth].max(), 
        pdf_values=list_cwrf_pdf[imonth], 
        border_color='lightcoral'
    )

# (Optional) Add labels
ax.set_xlabel("Target Month")
ax.set_ylabel("Temperature (F)")

plt.savefig(f'{init_year}{init_month:02}{init_day:02}_{vname}_PDF.png', dpi=150)
# plt.show()