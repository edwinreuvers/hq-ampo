import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.transforms as mtransforms
from matplotlib.patches import Arc, RegularPolygon
import subprocess
from pathlib import Path

#%%
def style(plt,**kwargs):
    style = plt.rcParamsDefault.copy()
    
    # Set default values for optional inputs
    fontname    = kwargs.get('fontname', 'lmodern')        # Default to 'lmodern' if 'fontname' is not provided
    fontsize    = kwargs.get('fontsize', 11)               # Default to 11 if 'fontsize' is not provided
    grid        = kwargs.get('grid', False) 
        
    # Standard style
    style_ax = {
        'lines.linestyle': "-",
        'lines.linewidth': 1,
        # Box off
        'axes.spines.bottom': True,
        'axes.spines.left': True,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'legend.frameon': False,
        # legend tight
        'legend.borderpad': 0,
        # 'axes.autoscale.tight': True, ????
        # 'savefig.bbox_inches': 'tight',????
        'axes.axisbelow': True,
    }
     
    # Fontsize
    style_ftsize = {
        "font.size": fontsize,                    # Set font size
        "legend.fontsize": 'small',              # Set legend font size
        "legend.title_fontsize": fontsize,        # Set legend title font size
        "axes.titlesize": fontsize,               # Set title font size
        }
    
    if fontname in ('libertinus', 'Libertinus'):
        style_ftname = {
            "text.usetex": True,
            "text.latex.preamble": "\n".join([
                r"\usepackage{libertine}",
                r"\usepackage[libertine]{newtxmath}",
                r"\usepackage[scale=0.9181263229]{roboto}",
                r'\def\mathdefault{\textsf}',
                r'\usepackage{xcolor}'
                ])}
    elif fontname in ('Minion Pro', 'MinionPro', 'minion pro', 'minionpro'):     
        style_ftname = {
            "font.family": "Myriad Pro",
            'mathtext.fontset': 'custom',
            'mathtext.rm': 'Minion Pro',
            'mathtext.it': 'Minion Pro:italic',
            'mathtext.bf': 'Minion Pro:bold',
            'mathtext.sf': 'Minion Pro',
            'mathtext.cal': 'Minion Pro',
            "text.usetex": False,
            "text.latex.preamble": "\n".join([
                r"\usepackage{MinionPro}",
                r"\usepackage{MyriadPro}",
                r'\def\mathdefault{\textsf}'
                r"\renewcommand{\rmdefault}{\sfdefault}",  # default text is sans
                r"\usepackage{amsmath}"  # Needed for \text{} in math mode
                ]),
            }
    elif fontname in ('lmodern', 'Latin Modern', 'latin modern'):
        style_ftname = {
            "text.usetex": True,
            'text.latex.preamble': r"""
                \usepackage{lmodern}            % Load the Latin Modern fonts
                \renewcommand{\familydefault}{\sfdefault}  % Set the default font to sans-serif
                \usepackage{amsmath}            % For math environments
            """,
            "font.family": "sans-serif",        # Set font family to sans-serif
        }
    
    if grid == 'on' or grid == True:
        style_grid = {
            # Define nice grid
            'axes.grid': True,
            'grid.alpha': 1,
            'grid.color': '#ccc', # e5e5e5 = 90% ccc=80%
            'grid.linestyle': '--',
            'grid.linewidth': 1,
            # # Minor grid
            'axes.grid.which': 'both',  # Show both major and minor grids
            # xtick/ytick
            'xtick.direction': 'in',
            'xtick.major.size': 0.0,
            'xtick.minor.size': 0.0,
            'ytick.direction': 'in',
            'ytick.major.size': 0.0,
            'ytick.minor.size': 0.0,
            }
    else: 
        style_grid = {} 
    
    style.update({**style_ax, **style_ftsize, **style_ftname, **style_grid})
    plt.rcParams.update(style)
    return None

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def add_labels(fig, axs, labels, x_offset=-3/72, y_offset=3/72, location='ax', fontsize='medium', fontweight='bold', usetex=False):
    """
    Add labels to a list of axes.

    Inputs:
    -----------
    fig : matplotlib.figure.Figure
        The figure object.
    axs : list of matplotlib.axes.Axes
        The axes to label.
    labels : list of str
        The labels to add.
    location : str, 'ax' or 'fig'
        Whether to place the labels relative to each axes ('ax') or relative to the figure ('fig').
    x_offset, y_offset : float
        Offsets in inches (converted to figure coordinates) to adjust label position.
    fontsize : str or int
        Font size of the labels.
    fontweight : str or int
        Font weight of the labels.
    """
    
    raw_labels = labels
    if usetex == True:
        labels = [rf'\textbf{{{label}}}' for label in raw_labels]
    
    for ax, label, raw_label in zip(axs, labels, raw_labels):
        # Create a small translation in inches
        trans = mtransforms.ScaledTranslation(x_offset, y_offset, fig.dpi_scale_trans)
        
        if location == 'ax':
            # Place relative to the axis
            ax.text(0, 1, label, transform=ax.transAxes + trans,
                    ha='right', va='center', fontsize=fontsize, fontweight=fontweight,
                    clip_on=False, usetex=usetex)
        elif location == 'fig':
            # Place relative to the figure
            fig.text(0, 1, label, transform=ax.transAxes + trans,
                     ha='right', va='center', fontsize=fontsize, fontweight=fontweight, usetex=usetex)
        else:
            raise ValueError("location must be 'ax' or 'fig'")
        ax.label = raw_label


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncate a colormap to a subset of its range.

    Inputs
    ----------
    cmap : matplotlib.colors.Colormap
        Colormap name or Colormap instance to truncate.
    minval : float, optional
        Minimum value (0 to 1) of the colormap to keep. Default is 0.0.
    maxval : float, optional
        Maximum value (0 to 1) of the colormap to keep. Default is 1.0.
    n : int, optional
        Number of points in the new colormap. Default is 100.

    Outputs
    -------
    matplotlib.colors.LinearSegmentedColormap
        Truncated colormap.
    """
       
    # Create the truncated colormap
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    
    return new_cmap

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def report_axes_size(fig, axs):
    """Print axis size in cm and aspect ratio."""
    
    for ax in axs.flatten():
        w, h = fig.get_size_inches()
        bbox = ax.get_position()
        w_cm = bbox.width * w * 2.54
        h_cm = bbox.height * h * 2.54
    
        try: 
            print(f"Panel {ax.label}, axis size (WxH): {w_cm:.3f}×{h_cm:.3f} cm, ratio={w_cm/h_cm:.3f}")
        except:
            print(f"Axis size: {w_cm:.3f}×{h_cm:.3f} cm  (ratio={w_cm/h_cm:.3f})")
            
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def report_fig_size(filepath,dpi=600):
    """Load figure and print pixel size, if exists."""
    
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()[1:]  # remove the dot and lowercase
    
    try:
        if ext == 'png':
        
            from PIL import Image
            img = Image.open(filepath)
            width_px, height_px = img.size
            width = width_px / dpi * 2.54
            height = height_px / dpi * 2.54
        
        elif ext == 'svg':
            import xml.etree.ElementTree as ET
             # Laad het SVG-bestand
            tree = ET.parse(filepath)
            root = tree.getroot()
             
            width_attr = root.attrib.get("width", None)
            height_attr = root.attrib.get("height", None)
            
            def convert_to_cm(value):
                if value is None:
                    return None
                value = value.strip()
                if value.endswith("cm"):
                    return float(value.replace("cm",""))
                elif value.endswith("mm"):
                    return float(value.replace("mm","")) / 10
                elif value.endswith("in"):
                    return float(value.replace("in","")) * 2.54
                elif value.endswith("pt"):
                    return float(value.replace("pt","")) * 2.54 / 72
                elif value.endswith("px"):
                    return float(value.replace("px","")) * 2.54 / 96
                else:
                    # assume pixels
                    return float(value) * 2.54 / 96
            
            width = convert_to_cm(width_attr)
            height = convert_to_cm(height_attr)
            
            # fallback: use viewBox if width/height not specified
            if width is None or height is None:
                viewbox = root.attrib.get("viewBox", None)
                if viewbox:
                    _, _, w, h = map(float, viewbox.split())
                    width = width or w * 2.54 / 96
                    height = height or h * 2.54 / 96
            
        elif ext == 'pdf':
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            page = reader.pages[0]
            width_pt = float(page.mediabox.width)
            height_pt = float(page.mediabox.height)
            width = width_pt * 2.54 / 72
            height = height_pt * 2.54 / 72
            
        else:
            print(f"Unsupported file format: {ext}")
            return

        print(f"{ext:} Fig dimenisions (WxH): {width:0.3f} x {height:0.3f} cm")
            
    except Exception as e:
        print(f"Could not load image '{filepath}':", e)
        

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def latex_has_package(pkg: str) -> bool:
    """Return True if LaTeX can load the given package."""
    tex_snippet = f"\\documentclass{{article}}\n\\usepackage{{{pkg}}}\n\\begin{{document}}OK\\end{{document}}"
    tmpfile = Path("testpkg.tex")
    tmpfile.write_text(tex_snippet)

    try:
        subprocess.run(
            ["latex", "-interaction=nonstopmode", str(tmpfile)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return (Path("testpkg.log").read_text().find("LaTeX Error") == -1)
    except Exception:
        return False
    finally:
        for ext in [".aux", ".log", ".tex", ".dvi"]:
            Path(f"testpkg{ext}").unlink(missing_ok=True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def find_font(filename):
    # Search common font directories on Windows
    possible_dirs = [
        r"C:\Windows\Fonts",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")
    ]
    
    for d in possible_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

#%%%%
def draw_circle(ax, radius, centX, centY, angle, theta2, color='black',
             direction='ccw', arrow_dir='along', shrink=0.0):
    """
    Draw an arc with an arrow on a matplotlib axis.
    
    Parameters:
    ax : matplotlib.axes.Axes
    radius : float
        Radius of the arc
    centX, centY : float
        Center coordinates
    angle : float
        Rotation of the arc (radians)
    theta2 : float
        Arc angular span (radians)
    color : str
        Color of the arc and arrow
    direction : str
        'ccw' or 'cw' for arc direction
    arrow_dir : str
        'along', 'opposite', 'cw', 'ccw'
    shrink : float
        Reduce the arc by this many radians (so arrow doesn't hit target)
    """

    # Adjust arc length
    if direction == 'ccw':
        theta2_adj = max(theta2 - shrink, 0)
        start_deg = 0
        end_deg = np.degrees(theta2_adj)
    else:  # cw
        theta2_adj = max(theta2 - shrink, 0)
        start_deg = np.degrees(theta2_adj)
        end_deg = 0

    # Arc start/end in degrees
    start_deg = 0
    end_deg = np.degrees(theta2_adj)
    if direction == 'cw':
        start_deg, end_deg = end_deg, start_deg

    # Draw arc
    arc = Arc([centX, centY], radius, radius,
              angle=np.degrees(angle),
              theta1=start_deg,
              theta2=end_deg,
              capstyle='round',
              linestyle='-',
              lw=1,
              color=color)
    ax.add_patch(arc)

    # Compute arrow position at end of shortened arc
    end_angle = angle + (theta2_adj if direction == 'ccw' else 0)
    endX = centX + (radius/2) * np.cos(end_angle)
    endY = centY + (radius/2) * np.sin(end_angle)

    # Tangent vector for arrow orientation
    if direction == 'ccw':
        tx, ty = -np.sin(end_angle), np.cos(end_angle)
    else:
        tx, ty = np.sin(end_angle), -np.cos(end_angle)
    arrow_angle = np.arctan2(ty, tx)

    # Adjust orientation based on arrow_dir
    if arrow_dir == 'opposite':
        arrow_angle += np.pi
    elif arrow_dir == 'cw':
        arrow_angle = 0
    elif arrow_dir == 'ccw':
        arrow_angle = np.pi/2
    # 'along' uses tangent angle
    
    arrow_angle = angle+theta2
    
    # Draw arrowhead
    ax.add_patch(
        RegularPolygon(
            (endX, endY),
            3,
            radius=0.01,
            orientation=arrow_angle,
            color=color
        )
    )