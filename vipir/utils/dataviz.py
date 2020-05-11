import numpy as np

import matplotlib.patches as patches
import matplotlib.ticker as ticker

def draw_box_on_axis(ax, l_left, xext, yext,
                    label=None, color='red', linestyle='-',linewidth=3):

    r = patches.Rectangle( l_left, xext, yext,facecolor='none',
                      edgecolor= color, linestyle=linestyle, linewidth=linewidth, label=label)

    ax.add_patch(r)
    

def show_objects(vip, objs, which='total_power', ann=None, ax=None, iou=None, colorbar=True,
                 title = None,
                 crop=True):
    @ticker.FuncFormatter
    def major_formatter_f(x, pos):
        return "%.2f" % (x/1000.0)
    
    @ticker.FuncFormatter
    def major_formatter_r(x, pos):
        return "%.2f" % x
    
    pcm = ax.pcolormesh(vip.freq, vip.rng, vip.snr(which=which).T, vmin=0, vmax=60, cmap='gnuplot')
    
    label_str = None # if this gets a value, show the legend
    
    if len(objs) >0:
        for o in objs:
            rngfreq = o['rngfreq']['coords']
            label_str = 'Detected Class: '+ o['label']
            draw_box_on_axis(ax, (rngfreq['minfreq'], rngfreq['minrng']), #origin
                                  rngfreq['maxfreq'] - rngfreq['minfreq'], #x-extent
                                  rngfreq['maxrng'] - rngfreq['minrng'], #y-extent
                                  label=label_str)
            
    #put on the ground truth
    if ann is not None:
        true_bbox = ann['bndbox']
        label_str = 'Ground Truth: '+ann['name']
        rngfreq = vip.bbox_rngfreq(true_bbox)['coords']
        draw_box_on_axis(ax, (rngfreq['minfreq'], rngfreq['minrng']), #origin
                              rngfreq['maxfreq'] - rngfreq['minfreq'], #x-extent
                              rngfreq['maxrng'] - rngfreq['minrng'],   #y-extent
                              linestyle=':', color = 'white', label=label_str)
    
    if iou is not None:
        textstr = r'$\mathrm{iou}=%.4f$' % iou
        props = dict(boxstyle='round', facecolor='lightgrey') #, alpha=0.5
        ax.text(0.80, 0.80, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    if type(title) == str:
        ax.set_title(title, fontdict={'size':16, 'weight':'bold'})
    elif title==True:
        ax.set_title(f'Station: {vip.station}\nDateTime: {vip.obs_time}',
                    fontdict={'size':16, 'weight':'bold'})
    ax.set_ylabel('Range (km)')
    ax.set_xlabel('Frequency (MHz)')

    ax.set_xscale('log')
 
    if crop:
        ax.set_yscale('log')
        ax.set_ylim(50,None)
        
    ax.grid()
    if label_str is not None:
        ax.legend(loc='upper right', fontsize=14)

    ax.set_xticks(np.logspace(np.log10(vip.minfreq), np.log10(vip.maxfreq),8))
    ax.xaxis.set_major_formatter(major_formatter_f)
    
    if crop:
        ax.set_yticks(np.logspace(np.log10(50), np.log10(1000),8))
    #ax.set_yticks(np.log10(np.array([50,70,100,140,200,300,500,900])))
    ax.yaxis.set_major_formatter(major_formatter_r)

    if colorbar:
        cbar=fig.colorbar(pcm, ax=ax)
        cbar.set_label('Total Power\nSignal to Noise Ratio (dB)', rotation=270, labelpad=30)
        
    return pcm