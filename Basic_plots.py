#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:38:32 2023

@author: oliverlemke
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
from statsmodels.stats.multitest import multipletests

def plot_hist(data_frame, keys, outfile="Out_hist.pdf", x_label="x", y_label="y", fs=15, n_bins=20, smoothing_factor=0.01, legend_loc="upper left", x_lim=None, grid=True):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)

    max_y = 0
    
    if not x_lim:
        x_lim = (np.nanmin(data_frame[keys.keys()].values)-(0.05*np.nanmax(data_frame[keys.keys()].values)),np.nanmax(data_frame[keys.keys()].values)+(0.05*np.nanmax(data_frame[keys.keys()].values)))
    
    for key in keys:

        hist = np.histogram(data_frame[key].values, range=x_lim, bins=n_bins, density=True)
        x = (hist[1][1:]+hist[1][:-1])/2

        spl = UnivariateSpline(np.insert(x,len(x),x_lim[1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
        spl.set_smoothing_factor(smoothing_factor)
        
        xs = np.linspace(x_lim[0],x_lim[1],1000)
        
        ax.plot(xs, spl(xs), color=keys[key]["Color"], label=keys[key]["Label"])
        #ax.plot(x, hist[0], color=keys[key]["Color"])
        ax.fill_between(xs, np.zeros(len(xs)), spl(xs), color=keys[key]["Color"], alpha=.6)
        
        max_y = np.nanmax((max_y,np.nanmax(hist[0])))
    
    if grid:
        ax.grid(axis='both', color='0.8')
           
    ax.set_xlim(x_lim)           
    ax.set_ylim(0,max_y*1.1)

    ax.set_xlabel(x_label, fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs)
    
    try:
        legend = plt.legend(loc=legend_loc, fontsize=fs, shadow=True, fancybox=True, framealpha=1)
    except:
        print("legend_loc not found. Using upper left as a default.")
        legend = plt.legend(loc="upper left", fontsize=fs, shadow=True, fancybox=True, framealpha=1)
    legend.get_frame().set_linewidth(2)

    for legobj in legend.legendHandles:
        legobj.set_linewidth(5)

    plt.tight_layout()
    plt.savefig(outfile)
    
    
def plot_correlation_scatter(data_frame, keys, outfile="Out_correlation_scatter.pdf", fs=20, fs_text=15, n_bins=20, smoothing_factor=1e-10, text_loc="lower right", color = "C0", pearson = True, spearman = True, p_pearson = None, p_spearman = None, x_lim = None, y_lim = None, plot_linreg = True, plot_xy = False, grid = True):

    # Add second layer (alpha=1, color_second_layer="C1", hist second layer normalized to max of everything)    
    # Separate fontsize for labels and legend
    # Formatter for same precision labels
    
    fig = plt.figure()
    fig.set_size_inches(7.5,7.5)
    gs = gridspec.GridSpec(2,2,width_ratios=[10,2],height_ratios=[2,10], wspace=0.01, hspace=0.01)
    
    key_x = list(keys.keys())[0]
    key_y = list(keys.keys())[1]
    
    data_to_plot = data_frame.copy()[[key_x,key_y]]
    data_to_plot.dropna(inplace=True)
    
    ax_scatter = fig.add_subplot(gs[1,0])
    ax_scatter.scatter(data_to_plot[key_x],data_to_plot[key_y],facecolors="None",edgecolors=color, marker=".",s=40)#,alpha=.6)
    
    if pearson:
        pearson_corr = pearsonr(data_to_plot[key_x],data_to_plot[key_y])
    else:
        pearson_corr = None
    if spearman:
        spearman_corr = spearmanr(data_to_plot[key_x],data_to_plot[key_y])
    else:
        spearman_corr = None
        
    ax_scatter.set_xlabel(keys[key_x]["Label"], fontsize=fs)
    ax_scatter.set_ylabel(keys[key_y]["Label"], fontsize=fs)
    ax_scatter.tick_params(axis="both", labelsize=fs)
    
    if not x_lim:
        x_lim = (np.nanmin(data_to_plot[key_x])-(0.05*np.nanmax(data_to_plot[key_x])),np.nanmax(data_to_plot[key_x])+(0.05*np.nanmax(data_to_plot[key_x])))
    ax_scatter.set_xlim(x_lim)
        
    if not y_lim:
        y_lim = (np.nanmin(data_to_plot[key_y])-(0.05*np.nanmax(data_to_plot[key_y])),np.nanmax(data_to_plot[key_y])+(0.05*np.nanmax(data_to_plot[key_y])))
    ax_scatter.set_ylim(y_lim)    
        
    if plot_linreg:
        coef = np.polyfit(data_to_plot[key_x],data_to_plot[key_y],1)
        poly1d_fn = np.poly1d(coef)
        ax_scatter.plot(x_lim,poly1d_fn(x_lim),c="k")
        
    if plot_xy:
        ax_scatter.plot(x_lim, x_lim, ls=":", c="k")
        
    if grid:
        ax_scatter.grid(axis='both', color='0.8')
        
    if pearsonr or spearmanr:
        text = ""
        if pearson_corr:
            if p_pearson:
                text += "R_Pearson = {0:.2f}\np_Pearson = {1:.2e}".format(pearson_corr[0],p_pearson)            
            else:
                text += "R_Pearson = {0:.2f}\np_Pearson = {1:.2e}".format(pearson_corr[0],pearson_corr[1])
        if pearson_corr and spearman_corr:
            text +="\n"
        if spearman_corr:
            if p_spearman:
                text += "R_Spearman = {0:.2f}\np_Spearman = {1:.2e}".format(spearman_corr[0],p_spearman)
            else:
                text += "R_Spearman = {0:.2f}\np_Spearman = {1:.2e}".format(spearman_corr[0],spearman_corr[1])
    
    try:    
        anchored_text = AnchoredText(text, loc=text_loc, prop=dict(size=fs_text))
    except:
        print("text_loc not found. Using upper left as a default.")
        ax_scatter.add_artist(AnchoredText(text, loc="upper left"))
    
    anchored_text.patch.set_alpha(0.5)
    anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax_scatter.add_artist(anchored_text)
    
    #
    ax_hist_x = fig.add_subplot(gs[0,0])
    ax_hist_x.axis("off")
    
    hist = np.histogram(data_to_plot[key_x].values, range=x_lim, bins=n_bins, density=True)
    x = (hist[1][1:]+hist[1][:-1])/2
    
    spl = UnivariateSpline(np.insert(x,len(x),hist[1][-1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
    spl.set_smoothing_factor(smoothing_factor)
    
    xs = np.linspace(x_lim[0],x_lim[1],1000)
    
    ax_hist_x.plot(xs, spl(xs), color=keys[key_x]["Color"])
    ax_hist_x.fill_between(xs, np.zeros(len(xs)), spl(xs), color=keys[key_x]["Color"], alpha=.6)
    ax_hist_x.plot([np.nanmedian(data_to_plot[key_x]),np.nanmedian(data_to_plot[key_x])],[0,np.nanmax(hist[0])*1.1],c="k",ls="--")
    
    ax_hist_x.set_xlim(x_lim)           
    ax_hist_x.set_ylim(0,np.nanmax(hist[0])*1.1)
    
    #
    ax_hist_y = fig.add_subplot(gs[1,1])
    ax_hist_y.axis("off")
    
    hist = np.histogram(data_to_plot[key_y].values, range=y_lim, bins=n_bins, density=True)
    y = (hist[1][1:]+hist[1][:-1])/2
    
    spl = UnivariateSpline(np.insert(y,len(y),hist[1][-1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
    spl.set_smoothing_factor(smoothing_factor)
    
    ys = np.linspace(y_lim[0],y_lim[1],1000)
    
    ax_hist_y.plot(spl(ys),ys, color=keys[key_y]["Color"])
    ax_hist_y.fill_betweenx(ys,spl(ys),np.zeros(len(ys)), color=keys[key_y]["Color"], alpha=.6)
    ax_hist_y.plot([0,np.nanmax(hist[0])*1.1],[np.nanmedian(data_to_plot[key_y]),np.nanmedian(data_to_plot[key_y])],c="k",ls="--")
    
    ax_hist_y.set_ylim(y_lim)     
    ax_hist_y.set_xlim(0,np.nanmax(hist[0])*1.1)
    
    plt.savefig(outfile, bbox_inches="tight")

def plot_binned_average(data_list, N=100, outfile="Out_binned_average", cmap = "viridis", fs = 15, x_label = "rel. Index", y_label="Mean(y)", boundaries=None, y_lim=None, grid = True):

    data_grouped = np.asarray([[np.nanmean(item[int(np.ceil((n)*len(item)/N)):int(np.ceil((n+1)*len(item)/N))]) for n in range(N)] for item in data_list])
    mean_grouped = np.nanmean(data_grouped, axis=0)
    std_grouped = np.nanstd(data_grouped, axis=0)
    cv_grouped = std_grouped/mean_grouped

    try:
        cm = mpl.cm.get_cmap(cmap)
    except:
        raise ValueError("Color map not found")
        
    min_cv = np.min(cv_grouped)
    max_cv = np.max(cv_grouped)
    
    fig = plt.figure()
    fig.set_size_inches(6,4)
    gs = gridspec.GridSpec(1,2, width_ratios=[5,1])
    
    xs = np.arange((1/(2*N)),1+(1/(2*N)),1/N)
    
    ax = fig.add_subplot(gs[0,0])
    for ind in range(N-1):
        ax.plot([xs[ind],xs[ind+1]], [mean_grouped[ind],mean_grouped[ind+1]], lw=3, c=cm((((cv_grouped[ind]+cv_grouped[ind+1])/2)-min_cv)/(max_cv-min_cv)))
        ax.fill_between([xs[ind],xs[ind+1]], [mean_grouped[ind]-std_grouped[ind],mean_grouped[ind+1]-std_grouped[ind+1]],[mean_grouped[ind]+std_grouped[ind],mean_grouped[ind+1]+std_grouped[ind+1]], color=cm((((cv_grouped[ind]+cv_grouped[ind+1])/2)-min_cv)/(max_cv-min_cv)), alpha=0.6)
    
    if boundaries:
        for boundary in boundaries:
            ax.plot([0,1],[boundary,boundary],c="k",ls="--")
            
    ax.set_xlabel(x_label, fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    
    ax.set_xlim(0,1)
    if y_lim:
        ax.set_ylim(y_lim)
    
    ax.tick_params(axis="both",labelsize=fs)
    
    if grid:
        ax.grid(axis='both', color='0.8')
        
    cax = fig.add_subplot(gs[0,1])
    cax.matshow(np.linspace(min_cv,max_cv,101).reshape(-1,1), cmap=cm, aspect=0.1)
    
    cax.set_xticks([])
    
    cax.yaxis.tick_right()
    cax.set_yticks(np.linspace(0,100,6), [str(round(float(label), 2)) for label in (np.linspace(0,1,6)*(max_cv-min_cv))+min_cv], fontsize=fs)
    
    cax.set_ylabel("Coefficient of variation", fontsize=fs, rotation=90)
    
    plt.savefig(outfile, bbox_inches="tight")
    
def plot_correlations_heatmap(data_frame, keys, ref_keys, outfile="Out_correlation_heatmap.pdf", corr_type="Spearman", p_values=None, alpha=0.05, v_lim=None, fs=15):
    if corr_type == "Spearman":
        data_to_plot = np.asarray([[spearmanr(data_frame[key_1],data_frame[key_2])[0] for key_1 in keys] for key_2 in ref_keys])
        if not p_values:
            p_values = np.asarray([[spearmanr(data_frame[key_1],data_frame[key_2])[1] for key_1 in keys] for key_2 in ref_keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_vadjusted = p_values
    elif corr_type == "Pearson":
        data_to_plot = np.asarray([[pearsonr(data_frame[key_1],data_frame[key_2])[0] for key_1 in keys] for key_2 in ref_keys])
        if not p_values:
            p_values = np.asarray([[pearsonr(data_frame[key_1],data_frame[key_2])[1] for key_1 in keys] for key_2 in ref_keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_vadjusted = p_values
    else:
         raise ValueError("corr_type not found")   
    
    
    if not v_lim:
        v_lim = (-np.ceil(10*np.max((np.max(data_to_plot),np.abs(np.min(data_to_plot)))))/10,np.ceil(10*np.max((np.max(data_to_plot),np.abs(np.min(data_to_plot)))))/10)
    
    cmap = "seismic"
    
    #fig = plt.figure()
    fig, ax = plt.subplots()
    fig.set_size_inches((len(keys)/2)+1,(len(ref_keys)/2)+1)
    #gs = gridspec.GridSpec(2, 1)
    
    #ax = fig.add_subplot(gs[0,0])
    im = ax.matshow(data_to_plot, cmap=cmap, vmin=v_lim[0], vmax=v_lim[1])
    
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([keys[key]["Label"] for key in keys],rotation=90, fontsize=fs)
    
    ax.set_yticks(np.arange(len(ref_keys)))
    ax.set_yticklabels([ref_keys[key]["Label"] for key in ref_keys], fontsize=fs)
    
    ax.tick_params(axis="both", labelsize=fs)
    ax.tick_params(axis="x", bottom=False)
    
    for ind in range(len(keys)):
        for ind2 in range(len(ref_keys)):
            if p_adjusted[ind2,ind]<alpha:
                ax.plot(ind,ind2, c="k", marker=(8, 2, 0))
            
    cbar = plt.colorbar(im, orientation="horizontal", pad=0.1)
    cbar.set_label(corr_type+" correlation", fontsize=fs)
    cbar.ax.tick_params(axis="x", labelsize=fs)
    
    plt.savefig(outfile,bbox_inches="tight") 
    
def plot_correlations_grid(data_frame, keys, ref_key, outfile = "Out_correlation_grid,pdf", y_label = "y", n_columns = 4, fs = 15, n_rows = None, x_lim = None, y_lim = None, plot_linreg = True, legend_loc = "upper left"):
    
    if not n_rows:
        n_rows = int(np.ceil(len(keys)/n_columns))
        
    if n_columns * n_rows < len(keys):
        raise ValueError("Grid resolution not fitting a plots. Increase n_columns or n_rows.")
    
    key_x = list(ref_key.keys())[0]
    
    fig = plt.figure()
    fig.set_size_inches((n_columns*3),(n_rows*3)+1)
    gs = gridspec.GridSpec(n_rows, n_columns, wspace=0.03, hspace=0.03)
    
    if not x_lim:
        x_lim = (np.nanmin(data_frame[key_x])-(0.05*np.nanmax(data_frame[key_x])),np.nanmax(data_frame[key_x])+(0.05*np.nanmax(data_frame[key_x])))  
    
    if not y_lim:
        y_lim = (np.nanmin(data_frame[list(keys.keys())])-(0.05*np.nanmax(data_frame[list(keys.keys())])),np.nanmax(data_frame[list(keys.keys())])+(0.05*np.nanmax(data_frame[list(keys.keys())])))
          
    for ind,key_y in enumerate(keys):
        data_to_plot = data_frame.copy()[[key_x,key_y]]
        
        ax = fig.add_subplot(gs[int(np.floor(ind/n_columns)),int(np.mod(ind,n_columns))])        
        ax.scatter(data_to_plot[key_x],data_to_plot[key_y],facecolors="None",color=keys[key_y]["Color"], marker=".",s=40, label=keys[key_y]["Label"])
            
        if plot_linreg:
            coef = np.polyfit(data_to_plot[key_x],data_to_plot[key_y],1)
            poly1d_fn = np.poly1d(coef)
            ax.plot(x_lim,poly1d_fn(x_lim),c="k")
            
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        if (int(np.mod(ind,n_columns)) == 0) and (int(np.floor(ind/n_columns)) == 0):
            ax.set_xlabel(ref_key[key_x]["Label"], fontsize=fs)
            ax.set_ylabel(y_label, fontsize=fs)
            ax.xaxis.set_label_position('top')
        
        if (int(np.mod(ind,n_columns)) == 0):
            ax.tick_params(axis="y", labelsize=fs)
        else:
            ax.set_yticklabels([])
        if (int(np.floor(ind/n_columns)) == 0):
            ax.tick_params(axis="x", labelsize=fs)
            ax.xaxis.set_ticks_position('top')
        else:
            ax.set_xticklabels([])
        
        ax.grid(axis='both', color='0.8')
        
        try:
            legend = ax.legend(loc=legend_loc, fontsize=fs)
        except:
            print("legend_loc not found. Using upper left as a default.")
            legend = ax.legend(loc="upper left", fontsize=fs, shadow=True, fancybox=True)
            
        legend.get_frame().set_linewidth(2)

        for legobj in legend.legendHandles:
            legobj.set_sizes([100])
            legobj.set_linewidth(5)
            
    plt.savefig(outfile, bbox_inches="tight")
        
        
        
        
        
        
        
        
        
        