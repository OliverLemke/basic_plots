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
from scipy.stats import pearsonr, spearmanr, kendalltau, ranksums, wilcoxon
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_curve, roc_auc_score
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from matplotlib.patches import Patch

#%%
# TODO
## Error handling for missing keys

#%%

# Histogram
def plot_hist(data_frame, keys, outfile="Out_hist.pdf", x_label="x", y_label="y", fs=15, fs_legend=15, n_bins=20, smoothing_factor=1e-10, legend_loc="upper left", x_lim=None, y_lim=None, grid=True,density=True):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)

    max_y = 0
    
    if not x_lim:
        x_lim = (np.nanmin(data_frame[keys.keys()].values)-(0.05*np.nanmax(data_frame[keys.keys()].values)),np.nanmax(data_frame[keys.keys()].values)+(0.05*np.nanmax(data_frame[keys.keys()].values)))
    
    for key in keys:

        hist = np.histogram(data_frame[key].values, range=x_lim, bins=n_bins, density=density)
        x = (hist[1][1:]+hist[1][:-1])/2

        spl = UnivariateSpline(np.insert(x,len(x),x_lim[1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
        spl.set_smoothing_factor(smoothing_factor)
        
        xs = np.linspace(x_lim[0],x_lim[1],1000)
        
        ax.plot(xs, spl(xs), color="w",zorder=1)
        ax.fill_between(xs, np.zeros(len(xs)), spl(xs), color="w", alpha=1,zorder=1)
        
        ax.plot(xs, spl(xs), color=keys[key]["Color"], label=keys[key]["Label"],zorder=10)
        ax.fill_between(xs, np.zeros(len(xs)), spl(xs), color=keys[key]["Color"], alpha=.6,zorder=10)
        
        max_y = np.nanmax((max_y,np.nanmax(hist[0])))
    
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis='both', color='0.8')
           
    ax.set_xlim(x_lim)           
    ax.set_ylim(0,max_y*1.1)
    
    if y_lim:
        ax.set_ylim(y_lim)

    ax.set_xlabel(x_label, fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs)
    
    try:
        legend = plt.legend(loc=legend_loc, fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
    except:
        print("legend_loc not found. Using upper left as a default.")
        legend = plt.legend(loc="upper left", fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
    legend.get_frame().set_linewidth(2)
    legend.set_zorder(20)

    for legobj in legend.legendHandles:
        legobj.set_linewidth(5)

    plt.tight_layout()
    plt.savefig(outfile)
    
def plot_hist_selection(data_frame, selections, ref_key, outfile="Out_hist_selection.pdf", x_label="x", y_label="y", fs=15, fs_legend=15, n_bins=20, smoothing_factor=1e-10, legend_loc="upper left", x_lim=None, grid=True, plot_full=True):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6,4)

    max_y = 0
    key_x = list(ref_key.keys())[0]
    
    if not x_lim:
        x_lim = (np.nanmin(data_frame[key_x].values)-(0.05*np.nanmax(data_frame[key_x].values)),np.nanmax(data_frame[key_x].values)+(0.05*np.nanmax(data_frame[key_x].values)))
    
    for selection in selections:

        hist = np.histogram(data_frame.loc[selections[selection]["Indices"],key_x].values, range=x_lim, bins=n_bins, density=True)
        x = (hist[1][1:]+hist[1][:-1])/2

        spl = UnivariateSpline(np.insert(x,len(x),x_lim[1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
        spl.set_smoothing_factor(smoothing_factor)
        
        xs = np.linspace(x_lim[0],x_lim[1],1000)
        
        ax.plot(xs, spl(xs), color="w",zorder=1)
        ax.fill_between(xs, np.zeros(len(xs)), spl(xs), color="w", alpha=1,zorder=1)
        
        ax.plot(xs, spl(xs), color=selections[selection]["Color"], label=selections[selection]["Label"],zorder=10)
        ax.fill_between(xs, np.zeros(len(xs)), spl(xs), color=selections[selection]["Color"], alpha=.6,zorder=10)
        
        max_y = np.nanmax((max_y,np.nanmax(hist[0])))
    
    if plot_full:
        hist = np.histogram(data_frame[key_x].values, range=x_lim, bins=n_bins, density=True)
        x = (hist[1][1:]+hist[1][:-1])/2

        spl = UnivariateSpline(np.insert(x,len(x),x_lim[1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
        spl.set_smoothing_factor(smoothing_factor)
        
        xs = np.linspace(x_lim[0],x_lim[1],1000)
        
        ax.plot(xs, spl(xs), color="w", alpha=1)
        ax.fill_between(xs, np.zeros(len(xs)), spl(xs), color="w", alpha=1)
        
        ax.plot(xs, spl(xs), color=ref_key[key_x]["Color"], label=ref_key[key_x]["Label"], alpha=.6)
        ax.fill_between(xs, np.zeros(len(xs)), spl(xs), color=ref_key[key_x]["Color"], alpha=.4)
        
        max_y = np.nanmax((max_y,np.nanmax(hist[0])))
    
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis='both', color='0.8')
           
    ax.set_xlim(x_lim)           
    ax.set_ylim(0,max_y*1.1)

    ax.set_xlabel(x_label, fontsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs)
    
    try:
        legend = plt.legend(loc=legend_loc, fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
    except:
        print("legend_loc not found. Using upper left as a default.")
        legend = plt.legend(loc="upper left", fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
    legend.get_frame().set_linewidth(2)
    legend.set_zorder(20)

    for legobj in legend.legendHandles:
        legobj.set_linewidth(5)

    plt.tight_layout()
    plt.savefig(outfile)
    
def fit_1_over_x(x,a,b,c):
    return a/(x+b)+c

# Scatter    
def plot_correlation_scatter(data_frame, keys, outfile="Out_correlation_scatter.pdf", fs=20, fs_text=15, n_bins=20, smoothing_factor=1e-10, text_loc="lower right", color = "C0", pearson = False, spearman = False, kendall = False, p_pearson = None, p_spearman = None, p_kendall = None, x_lim = None, y_lim = None, plot_linreg = True, plot_xy = False, grid = True, highlight = None, highlight_label = False, legend_loc = "upper left", plot_1_over_x=False, highlight_size=150, round_p=False):

    # Formatter for same precision labels
    # Add title
    # Add gradient for highlighting
    ## Use "if gradient" before "if highlight", use cmap -> Include colorbar!!!!
    ### Use scatter(x,y,c=value,cmap)
    ### Include colorbar
    
    # Include dict of dicts and then put everything in one loop, only modify, remove correlation, add hist 
    
    fig = plt.figure()
    fig.set_size_inches(7.5,7.5)
    gs = gridspec.GridSpec(2,2,width_ratios=[10,2],height_ratios=[2,10], wspace=0.01, hspace=0.01)
    
    key_x = list(keys.keys())[0]
    key_y = list(keys.keys())[1]
    
    data_to_plot = data_frame.copy()[[key_x,key_y]]
    data_to_plot.dropna(inplace=True)
    
    ax_scatter = fig.add_subplot(gs[1,0])
    ax_scatter.scatter(data_to_plot[key_x],data_to_plot[key_y],facecolors="w",edgecolors="w", marker=".",s=40)
    ax_scatter.scatter(data_to_plot[key_x],data_to_plot[key_y],facecolors="None",edgecolors=color, marker=".",s=40)#,alpha=.6)
    
    if highlight:
        for key in highlight:
            if highlight_label:
                ax_scatter.scatter(data_to_plot.loc[highlight[key]["Indices"],key_x],data_to_plot.loc[highlight[key]["Indices"],key_y],facecolors=highlight[key]["Color"],edgecolors=highlight[key]["Color"], marker=".",s=highlight_size, label=key)#,alpha=.6)
            else:
                ax_scatter.scatter(data_to_plot.loc[highlight[key]["Indices"],key_x],data_to_plot.loc[highlight[key]["Indices"],key_y],facecolors=highlight[key]["Color"],edgecolors=highlight[key]["Color"], marker=".",s=highlight_size)

    if pearson:
        pearson_corr = pearsonr(data_to_plot[key_x],data_to_plot[key_y])
    else:
        pearson_corr = None
    if spearman:
        spearman_corr = spearmanr(data_to_plot[key_x],data_to_plot[key_y], nan_policy="omit")
    else:
        spearman_corr = None
    if kendall:
        kendall_corr = kendalltau(data_to_plot[key_x],data_to_plot[key_y], nan_policy="omit")
    else:
        kendall_corr = None
    
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
    
    if plot_1_over_x:
        popt, pcov = curve_fit(fit_1_over_x, data_to_plot[key_x], data_to_plot[key_y])
        xs = np.linspace(x_lim[0], x_lim[1],100)
        ax_scatter.plot(xs,fit_1_over_x(xs, *popt),c="k")
        
    if plot_xy:
        ax_scatter.plot(x_lim, x_lim, ls=":", c="k")
        
    if grid:
        ax_scatter.set_axisbelow(True)
        ax_scatter.grid(axis='both', color='0.8')
        
    if pearson or spearman or kendall:
        text = ""
        if pearson:
            if p_pearson:
                if round_p:
                    text += "PearsonR = {0:.2f}\np adj. < 1E{1:.0f}".format(pearson_corr[0],np.ceil(np.log10(p_pearson)))
                else:
                    text += "PearsonR = {0:.2f}\np adj. = {1:.2e}".format(pearson_corr[0],p_pearson)            
            else:
                if round_p:
                    text += "PearsonR = {0:.2f}\np < 1E{1:.0f}".format(pearson_corr[0],np.ceil(np.log10(pearson_corr[1])))
                else:
                    text += "PearsonR = {0:.2f}\np = {1:.2e}".format(pearson_corr[0],pearson_corr[1])
        if pearson and spearman:
            text +="\n"
        if spearman:
            if p_spearman:
                if round_p:
                    text += "SpearmanR = {0:.2f}\np adj. < 1E{1:.0f}".format(spearman_corr[0],np.ceil(np.log10(p_spearman)))
                else:
                    text += "SpearmanR = {0:.2f}\np adj. = {1:.2e}".format(spearman_corr[0],p_spearman)
            else:
                if round_p:
                    text += "SpearmanR = {0:.2f}\np < 1E{1:.0f}".format(spearman_corr[0],np.ceil(np.log10(spearman_corr[1])))
                else:
                    text += "SpearmanR = {0:.2f}\np = {1:.2e}".format(spearman_corr[0],spearman_corr[1])
        if (kendall and spearman) or (kendall and pearson):
            text +="\n"
        if kendall:
            if p_kendall:
                if round_p:
                    text += "KendallTau = {0:.2f}\np adj. < 1E{1:.0f}".format(kendall_corr[0],np.ceil(np.log10(p_kendall)))
                else:
                    text += "KendallTau = {0:.2f}\np adj. = {1:.2e}".format(kendall_corr[0],p_kendall)
            else:
                if round_p:
                    text += "KendallTau = {0:.2f}\np < 1E{1:.0f}".format(kendall_corr[0],np.ceil(np.log10(kendall_corr[1])))
                else:
                    text += "KendallTau = {0:.2f}\np = {1:.2e}".format(kendall_corr[0],kendall_corr[1])
        try:    
            anchored_text = AnchoredText(text, loc=text_loc, prop=dict(size=fs_text))
        except:
            print("text_loc not found. Using lower right as a default.")
            anchored_text = AnchoredText(text, loc="lower right", prop=dict(size=fs_text))
            #ax_scatter.add_artist(AnchoredText(text, loc="lower right"))
    
        anchored_text.patch.set_alpha(0.5)
        anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax_scatter.add_artist(anchored_text)
    
    if highlight:
        if highlight_label:
            try:
                ax_scatter.legend(loc=legend_loc, fontsize=fs_text)
            except:
                print("legend_loc not found. Using upper left as a default.")
                ax_scatter.legend(loc="upper_left", fontsize=fs_text)
        
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
    
    #if highlight_hist and highlight:
    #    for key in highlight:
    #        hist_high_n = np.histogram(data_to_plot.loc[highlight[key]["Indices"],key_x].values, range=x_lim, bins=n_bins, density=True)
    #        hist_high = (hist_high_n[0]/np.max(hist_high_n[0])*np.max(hist[0]),hist_high_n[1])
    #        spl = UnivariateSpline(np.insert(x,len(x),hist_high[1][-1]),np.insert(hist_high[0],len(hist_high[0]),hist_high[0][-1]))
    #        spl.set_smoothing_factor(smoothing_factor)
    #        ax_hist_x.plot(xs, spl(xs), color=highlight[key]["Color"])
    #        ax_hist_x.fill_between(xs, np.zeros(len(xs)), spl(xs), color=highlight[key]["Color"], alpha=.6)
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
    
def plot_correlation_scatter_split(data_frame, keys, selection, outfile="Out_correlation_scatter.pdf", fs=20, fs_text=15, n_bins=20, smoothing_factor=1e-10, x_lim = None, y_lim = None, plot_xy = False, grid = True, legend_loc = "upper left"):
     
    fig = plt.figure()
    fig.set_size_inches(7.5,7.5)
    gs = gridspec.GridSpec(2,2,width_ratios=[10,2],height_ratios=[2,10], wspace=0.01, hspace=0.01)
    
    key_x = list(keys.keys())[0]
    key_y = list(keys.keys())[1]
    
    data_to_plot = data_frame.copy()[[key_x,key_y]]
    data_to_plot.dropna(inplace=True)
    
    ax_scatter = fig.add_subplot(gs[1,0])
    ax_hist_x = fig.add_subplot(gs[0,0])
    ax_hist_y = fig.add_subplot(gs[1,1])
    ax_hist_x.axis("off")
    ax_hist_y.axis("off")
    
    ax_scatter.scatter(data_to_plot[key_x],data_to_plot[key_y],facecolors="w",edgecolors="w", marker=".",s=40)
    
    if not x_lim:
        x_lim = (np.nanmin(data_to_plot[key_x])-(0.05*np.nanmax(data_to_plot[key_x])),np.nanmax(data_to_plot[key_x])+(0.05*np.nanmax(data_to_plot[key_x])))
    ax_scatter.set_xlim(x_lim)
        
    if not y_lim:
        y_lim = (np.nanmin(data_to_plot[key_y])-(0.05*np.nanmax(data_to_plot[key_y])),np.nanmax(data_to_plot[key_y])+(0.05*np.nanmax(data_to_plot[key_y])))
    ax_scatter.set_ylim(y_lim)   
    
    hist_max_x = 0
    hist_max_y = 0
    
    for key_sel,select in selection.items():
        
        indices = select["Indices"]
        color = select["Color"]
        plot_linreg = select["LinReg"]
        plot_1_over_x = select["1_x"]
        label = select["Label"]
    
        ax_scatter.scatter(data_to_plot.loc[indices,key_x],data_to_plot.loc[indices,key_y],facecolors="None",edgecolors=color, marker=".",s=40, label=label)#,alpha=.6)
    
        if plot_linreg:
            coef = np.polyfit(data_to_plot.loc[indices,key_x],data_to_plot.loc[indices,key_y],1)
            poly1d_fn = np.poly1d(coef)
            ax_scatter.plot(x_lim,poly1d_fn(x_lim),c=color)
        
        if plot_1_over_x:
            popt, pcov = curve_fit(fit_1_over_x, data_to_plot.loc[indices,key_x], data_to_plot.loc[indices,key_y])
            xs = np.linspace(x_lim[0], x_lim[1],100)
            ax_scatter.plot(xs,fit_1_over_x(xs, *popt),c="k")
            
        hist = np.histogram(data_to_plot.loc[indices,key_x].values, range=x_lim, bins=n_bins, density=True)
        x = (hist[1][1:]+hist[1][:-1])/2
        
        spl = UnivariateSpline(np.insert(x,len(x),hist[1][-1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
        spl.set_smoothing_factor(smoothing_factor)
        
        xs = np.linspace(x_lim[0],x_lim[1],1000)
        
        ax_hist_x.plot(xs, spl(xs), color=color)
        ax_hist_x.fill_between(xs, np.zeros(len(xs)), spl(xs), color=color, alpha=.6)
        ax_hist_x.plot([np.nanmedian(data_to_plot.loc[indices,key_x]),np.nanmedian(data_to_plot.loc[indices,key_x])],[0,np.nanmax(hist[0])*100],c=color,ls="--")
        
        hist_max_x = np.max((np.nanmax(hist[0]),hist_max_x))
        
        hist = np.histogram(data_to_plot.loc[indices,key_y].values, range=y_lim, bins=n_bins, density=True)
        y = (hist[1][1:]+hist[1][:-1])/2
        
        spl = UnivariateSpline(np.insert(y,len(y),hist[1][-1]),np.insert(hist[0],len(hist[0]),hist[0][-1]))
        spl.set_smoothing_factor(smoothing_factor)
        
        ys = np.linspace(y_lim[0],y_lim[1],1000)
        
        ax_hist_y.plot(spl(ys),ys, color=color)
        ax_hist_y.fill_betweenx(ys,spl(ys),np.zeros(len(ys)), color=color, alpha=.6)
        ax_hist_y.plot([0,np.nanmax(hist[0])*100],[np.nanmedian(data_to_plot.loc[indices,key_y]),np.nanmedian(data_to_plot.loc[indices,key_y])],c=color,ls="--")
        
        hist_max_y = np.max((np.nanmax(hist[0]),hist_max_y))
        
    ax_scatter.set_xlabel(keys[key_x]["Label"], fontsize=fs)
    ax_scatter.set_ylabel(keys[key_y]["Label"], fontsize=fs)
    ax_scatter.tick_params(axis="both", labelsize=fs)
    ax_hist_x.set_xlim(x_lim)   
    ax_hist_x.set_ylim(0,hist_max_x*1.1)
    ax_hist_y.set_ylim(y_lim)     
    ax_hist_y.set_xlim(0,hist_max_y*1.1)
       
    if plot_xy:
        ax_scatter.plot(x_lim, x_lim, ls=":", c="k")
        
    if grid:
        ax_scatter.set_axisbelow(True)
        ax_scatter.grid(axis='both', color='0.8')
        
    try:
        ax_scatter.legend(loc=legend_loc, fontsize=fs_text)
    except:
        print("legend_loc not found. Using upper left as a default.")
        ax_scatter.legend(loc="upper_left", fontsize=fs_text)
    
    plt.savefig(outfile, bbox_inches="tight")
    
def plot_correlations_heatmap(data_frame, keys, ref_keys, outfile="Out_correlation_heatmap.pdf", corr_type="Spearman", p_values=None, alpha=0.05, v_lim=None, fs=15, cmap = "seismic", rotation=0, rotation_cb=0, rotation_cb_label=0, va="center", plot_cbar=True, invert_cbar=False, discrete=False, discrete_first=False):
    if corr_type == "Spearman":
        
        if discrete:
            keys_discrete = {}
            for key in discrete:
                keys_discrete.update({key:keys[key]})
                keys.pop(key,"None")
        
        data_to_plot = np.asarray([[spearmanr(data_frame.copy()[[key_1,key_2]].dropna()[key_1],data_frame.copy()[[key_1,key_2]].dropna()[key_2])[0] for key_1 in keys] for key_2 in ref_keys])
        if discrete:
            data_discrete = np.asarray([[kendalltau(data_frame.copy()[[key_1,key_2]].dropna()[key_1],data_frame.copy()[[key_1,key_2]].dropna()[key_2])[0] for key_1 in keys_discrete] for key_2 in ref_keys])
            if discrete_first:
                data_to_plot = np.concatenate((data_discrete,data_to_plot),axis=1)
            else:
                data_to_plot = np.concatenate((data_to_plot,data_discrete),axis=1)
        if not p_values:
            p_values = np.asarray([[spearmanr(data_frame.copy()[[key_1,key_2]].dropna()[key_1],data_frame.copy()[[key_1,key_2]].dropna()[key_2])[1] for key_1 in keys] for key_2 in ref_keys])
            if discrete:
                p_discrete = np.asarray([[kendalltau(data_frame.copy()[[key_1,key_2]].dropna()[key_1],data_frame.copy()[[key_1,key_2]].dropna()[key_2])[1] for key_1 in keys_discrete] for key_2 in ref_keys])
                if discrete_first:
                    p_values = np.concatenate((p_discrete,p_values),axis=1)
                else:
                    p_values = np.concatenate((p_values,p_discrete),axis=1)
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    elif corr_type == "Pearson":
        data_to_plot = np.asarray([[pearsonr(data_frame.copy()[[key_1,key_2]].dropna()[key_1],data_frame.copy()[[key_1,key_2]].dropna()[key_2])[0] for key_1 in keys] for key_2 in ref_keys])
        if not p_values:
            p_values = np.asarray([[pearsonr(data_frame.copy()[[key_1,key_2]].dropna()[key_1],data_frame.copy()[[key_1,key_2]].dropna()[key_2])[1] for key_1 in keys] for key_2 in ref_keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    else:
         raise ValueError("corr_type not found")   

    if discrete:
        keys.update(keys_discrete)
        if discrete_first:
            for key in discrete[::-1]:
                keys = {key: keys.pop(key), **keys}
    
    if not v_lim:
        v_lim = (-np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10,np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10)
    
    #fig = plt.figure()
    fig, ax = plt.subplots()
    fig.set_size_inches((len(keys)/2)+1,(len(ref_keys)/2)+1)
    #gs = gridspec.GridSpec(2, 1)
    
    #ax = fig.add_subplot(gs[0,0])
    im = ax.matshow(data_to_plot, cmap=cmap, vmin=v_lim[0], vmax=v_lim[1])
    
    #if invert_cbar:
    #    im2 = ax.matshow([[-1*data_to_plot[0,0]]], cmap = cmap+"_r", vmin =v_lim[0], vmax = v_lim[1])
    #    im2.set_visible(False)
    
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([keys[key]["Label"] for key in keys],rotation=90, fontsize=fs)
    
    ax.set_yticks(np.arange(len(ref_keys)))
    ax.set_yticklabels([ref_keys[key]["Label"] for key in ref_keys], fontsize=fs,rotation=rotation, va=va)
    
    ax.tick_params(axis="both", labelsize=fs)
    ax.tick_params(axis="x", bottom=False)
    
    for ind in range(len(keys)):
        for ind2 in range(len(ref_keys)):
            if p_adjusted[ind2,ind]<alpha:
                ax.plot(ind,ind2, c="k", marker=(8, 2, 0))
    
    if plot_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="10%", pad=0.05)
        
        #if invert_cbar:
        #    cbar = plt.colorbar(im2, orientation="horizontal",cax=cax)
        #    tks = cbar.get_ticks()
        #    cbar.set_ticklabels(tks[::-1])
        #    cbar.ax.tick_params(axis="x", labelsize=fs, rotation=rotation_cb)

        #else:
        cbar = plt.colorbar(im, orientation="horizontal",cax=cax)
        cbar.ax.tick_params(axis="x", labelsize=fs, rotation=rotation_cb)
        
        if discrete:
            cbar.set_label("Correlation", fontsize=fs, rotation=rotation_cb_label)
        else:
            cbar.set_label(corr_type+" correlation", fontsize=fs, rotation=rotation_cb_label)
            
    
    #plt.tight_layout()
    plt.savefig(outfile,bbox_inches="tight") 
    
def plot_correlations_heatmap_selection(data_frame, keys, selections, ref_key, outfile="Out_correlation_heatmap_selection.pdf", corr_type="Spearman", p_values=None, alpha=0.05, v_lim=None, fs=15, cmap = "seismic", rotation=0, rotation_cb=0, plot_cbar=True, ha="center"):
    # include kendall-tau
    if corr_type == "Spearman":
        data_to_plot = np.asarray([[spearmanr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[0] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column)] for key in keys])
        if not p_values:
            p_values = np.asarray([[spearmanr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[1] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column)] for key in keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    elif corr_type == "Pearson":
        data_to_plot = np.asarray([[pearsonr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[0] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column)] for key in keys])
        if not p_values:
            p_values = np.asarray([[pearsonr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[1] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column)] for key in keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    else:
         raise ValueError("corr_type not found") 
         
    if not v_lim:
        v_lim = (-np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10,np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10)
    
    fig, ax = plt.subplots()
    fig.set_size_inches((len(selections)/2)+1,(len(keys)/2)+1)
    
    im = ax.matshow(data_to_plot, cmap=cmap, vmin=v_lim[0], vmax=v_lim[1])
    
    ax.set_xticks(np.arange(len(selections)))
    ax.set_xticklabels([selections[selection] for selection in selections],rotation=rotation, fontsize=fs, ha=ha)
    
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([keys[key]["Label"] for key in keys], fontsize=fs)
    
    ax.tick_params(axis="both", labelsize=fs)
    ax.tick_params(axis="x", bottom=False)
    
    for ind in range(len(selections)):
        for ind2 in range(len(keys)):
            if p_adjusted[ind2,ind]<alpha:
                ax.plot(ind,ind2, c="k", marker=(8, 2, 0))
    
    if plot_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        
        cbar = plt.colorbar(im, orientation="vertical", cax=cax)
        cbar.set_label(corr_type+" correlation", fontsize=fs)
        cbar.ax.tick_params(axis="y", labelsize=fs, rotation=rotation_cb)
    
    #plt.tight_layout()
    plt.savefig(outfile,bbox_inches="tight") 
    
def plot_correlations_heatmap_selection_double(data_frame, keys, selections, key_2, outfile="Out_correlation_heatmap_selection.pdf", corr_type="Spearman", p_values=None, alpha=0.05, v_lim=None, fs=15, cmap = "seismic", rotation=0, rotation_cb=0, rotation_cb_label=0, va="center", discrete=None, discrete_first=False, plot_cbar=True):
    
    if corr_type == "Spearman":
        
        if discrete:
            keys_discrete = {}
            for key in discrete:
                keys_discrete.update({key:keys[key]})
                keys.pop(key,"None")
        
        data_to_plot = np.asarray([[spearmanr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[0] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column) for ref_key in data_frame.columns if re.search(selection,ref_key) and re.search(key_2,ref_key)] for key in keys])
        if discrete:
            data_discrete = np.asarray([[kendalltau(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[0] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column) for ref_key in data_frame.columns if re.search(selection,ref_key) and re.search(key_2,ref_key)] for key in keys_discrete])
            if discrete_first:
                data_to_plot = np.concatenate((data_discrete,data_to_plot),axis=0)
            else:
                data_to_plot = np.concatenate((data_to_plot,data_discrete),axis=0)
        if not p_values:
            p_values = np.asarray([[spearmanr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[1] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column) for ref_key in data_frame.columns if re.search(selection,ref_key) and re.search(key_2,ref_key)] for key in keys])
            if discrete:
                p_discrete = np.asarray([[kendalltau(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[1] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column) for ref_key in data_frame.columns if re.search(selection,ref_key) and re.search(key_2,ref_key)] for key in keys_discrete])
                if discrete_first:
                    p_values = np.concatenate((p_discrete,p_values),axis=0)
                else:
                    p_values = np.concatenate((p_values,p_discrete),axis=0)
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    elif corr_type == "Pearson":
        data_to_plot = np.asarray([[pearsonr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[0] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column) for ref_key in data_frame.columns if re.search(selection,ref_key) and re.search(key_2,ref_key)] for key in keys])
        if not p_values:
            p_values = np.asarray([[pearsonr(data_frame.copy()[[column,ref_key]].dropna()[column],data_frame.copy()[[column,ref_key]].dropna()[ref_key])[1] for selection in selections for column in data_frame.columns if re.search(selection,column) and re.search(key,column) for ref_key in data_frame.columns if re.search(selection,ref_key) and re.search(key_2,ref_key)] for key in keys])
            p_adjusted = multipletests(p_values.reshape(-1),alpha=alpha,method="fdr_bh")[1].reshape(np.shape(p_values))
        else:
            p_adjusted = p_values
    else:
         raise ValueError("corr_type not found") 
    
    if discrete:
        keys.update(keys_discrete)
        if discrete_first:
            for key in discrete[::-1]:
                keys = {key: keys.pop(key), **keys}
    
    if not v_lim:
        v_lim = (-np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10,np.ceil(10*np.nanmax(np.abs(data_to_plot)))/10)
    
    fig, ax = plt.subplots()
    fig.set_size_inches((len(selections)/2)+1,(len(keys)/2)+1)
    
    im = ax.matshow(data_to_plot, cmap=cmap, vmin=v_lim[0], vmax=v_lim[1])
    
    ax.set_xticks(np.arange(len(selections)))
    ax.set_xticklabels([selections[selection] for selection in selections],rotation=90, fontsize=fs)
    
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels([keys[key]["Label"] for key in keys], fontsize=fs, rotation=rotation, va=va)
    
    ax.tick_params(axis="both", labelsize=fs)
    ax.tick_params(axis="x", bottom=False)
    
    for ind in range(len(selections)):
        for ind2 in range(len(keys)):
            if p_adjusted[ind2,ind]<alpha:
                ax.plot(ind,ind2, c="k", marker=(8, 2, 0))
    
    if plot_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        
        cbar = plt.colorbar(im, orientation="vertical", cax=cax)
        if discrete:
            cbar.set_label("Correlation", fontsize=fs, rotation=rotation_cb_label)
        else:
            cbar.set_label(corr_type+" correlation", fontsize=fs, rotation=rotation_cb_label)
        cbar.ax.tick_params(axis="y", labelsize=fs, rotation=rotation_cb)
    
    #plt.tight_layout()
    plt.savefig(outfile,bbox_inches="tight") 
    
def plot_correlations_grid(data_frame, keys, ref_key, outfile = "Out_correlation_grid,pdf", y_label = "y", n_columns = 4, fs = 15, n_rows = None, x_lim = None, y_lim = None, plot_linreg = True, legend_loc = "upper left"):
    
    if not n_rows:
        n_rows = int(np.ceil(len(keys)/n_columns))
        
    if n_columns * n_rows < len(keys):
        raise ValueError("Grid resolution not fitting all plots. Increase n_columns or n_rows.")
    
    key_x = list(ref_key.keys())[0]
    
    fig = plt.figure()
    fig.set_size_inches((n_columns*3)+1.2,(n_rows*3)+0.9)
    gs = gridspec.GridSpec(n_rows+1, n_columns+1, wspace=0.03, hspace=0.03, width_ratios=[4]+[10]*n_columns, height_ratios=[3]+[10]*n_rows)
    
    if not x_lim:
        x_lim = (np.nanmin(data_frame[key_x])-(0.05*np.nanmax(data_frame[key_x])),np.nanmax(data_frame[key_x])+(0.05*np.nanmax(data_frame[key_x])))  
    
    if not y_lim:
        y_lim = (np.nanmin(data_frame[list(keys.keys())])-(0.05*np.nanmax(data_frame[list(keys.keys())])),np.nanmax(data_frame[list(keys.keys())])+(0.05*np.nanmax(data_frame[list(keys.keys())])))
          
    for ind,key_y in enumerate(keys):
        data_to_plot = data_frame.copy()[[key_x,key_y]]
        
        ax = fig.add_subplot(gs[int(np.floor(ind/n_columns))+1,int(np.mod(ind,n_columns))+1])  
        ax.scatter(data_to_plot[key_x],data_to_plot[key_y],facecolors="w",color="w", marker=".",s=40)
        ax.scatter(data_to_plot[key_x],data_to_plot[key_y],facecolors="None",color=keys[key_y]["Color"], marker=".",s=40, label=keys[key_y]["Label"])
            
        if plot_linreg:
            coef = np.polyfit(data_to_plot[key_x],data_to_plot[key_y],1)
            poly1d_fn = np.poly1d(coef)
            ax.plot(x_lim,poly1d_fn(x_lim),c="k")
            
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        ax.tick_params(axis="y", labelsize=fs)
        ax.tick_params(axis="x", labelsize=fs)
        ax.xaxis.set_ticks_position('top')
        
        if not (int(np.mod(ind,n_columns)) == 0):
            ax.set_yticklabels([])
        if not (int(np.floor(ind/n_columns)) == 0):
            ax.set_xticklabels([])
        
        ax.set_axisbelow(True)
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
            
    ax_x = fig.add_subplot(gs[0,1:])
    ax_x.set_xlim(0,1)
    ax_x.set_ylim(0,1)
    ax_x.axis('off')
    
    ax_x.text(0.5,1.0,ref_key[key_x]["Label"], fontsize=fs, horizontalalignment='center', verticalalignment='top')
    
    ax_y = fig.add_subplot(gs[1:,0])
    ax_y.set_xlim(0,1)
    ax_y.set_ylim(0,1)
    ax_y.axis('off')
  
    ax_y.text(0.0,0.5,y_label, fontsize=fs, rotation=90, horizontalalignment='left', verticalalignment='center')
    
    plt.savefig(outfile, bbox_inches="tight")

def plot_correlations_boxplot(data_frame, keys, ref_key, ind_key=0, selection=None, corr_type="Spearman", outfile="Out_boxplot.pdf", fs=15, y_lim=None, grid=True):

    key_x = list(ref_key.keys())[ind_key]
    
    if selection:
        if corr_type=="Spearman":
            collection_R = [[spearmanr(data_frame.copy()[[key_x,column]].dropna()[key_x],data_frame.copy()[[key_x,column]].dropna()[column])[0] for column in data_frame.columns if re.findall(key,column) for el in selection if re.findall(el,column)] for key in keys]
        elif corr_type=="Pearson":
            collection_R = [[pearsonr(data_frame.copy()[[key_x,column]].dropna()[key_x],data_frame.copy()[[key_x,column]].dropna()[column])[0] for column in data_frame.columns if re.findall(key,column) for el in selection if re.findall(el,column)] for key in keys]            
        else:
            raise ValueError("Correlation type not defined.")    
    else:
        if corr_type=="Spearman":
            collection_R = [[spearmanr(data_frame.copy()[[key_x,column]].dropna()[key_x],data_frame.copy()[[key_x,column]].dropna()[column])[0] for column in data_frame.columns if re.findall(key,column)] for key in keys]
        elif corr_type=="Pearson":
            collection_R = [[pearsonr(data_frame.copy()[[key_x,column]].dropna()[key_x],data_frame.copy()[[key_x,column]].dropna()[column])[0] for column in data_frame.columns if re.findall(key,column)] for key in keys]
        else:
            raise ValueError("Correlation type not defined.")
    
    
    fig,ax = plt.subplots()
    fig.set_size_inches(len(keys)/3+1,5)
    bplot = ax.boxplot(collection_R,patch_artist=True,medianprops=dict(color="k"))
    
    colors = [keys[key]["Color"] for key in keys]
    for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    
    ax.plot([0.5,20.5],[0,0],c="k",ls=":")
    
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis='both', color='0.8')
    
    if not y_lim:
        y_lim=((-1)*np.nanmax(np.abs(collection_R))-0.05*np.nanmax(np.abs(collection_R)),np.nanmax(np.abs(collection_R))+0.05*np.nanmax(np.abs(collection_R)))
    
    ax.set_xticklabels([keys[key]["Label"] for key in keys],rotation=90, fontsize=fs)
    ax.set_ylim(y_lim)
    ax.set_ylabel(corr_type+"R "+ref_key[key_x]["Label"], fontsize=fs)
    
    ax.tick_params(axis="y",labelsize=fs)
    
    plt.savefig(outfile, bbox_inches="tight")

# Average
def plot_binned_average(data_list, N=100, outfile="Out_binned_average", cmap = "viridis", fs = 15, x_label = "rel. Index", y_label="Mean(y)", boundaries=None, y_lim=None, grid = True, return_data=False):

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
        ax.plot([xs[ind],xs[ind+1]], [mean_grouped[ind],mean_grouped[ind+1]], lw=3, c="w")
        ax.fill_between([xs[ind],xs[ind+1]], [mean_grouped[ind]-std_grouped[ind],mean_grouped[ind+1]-std_grouped[ind+1]],[mean_grouped[ind]+std_grouped[ind],mean_grouped[ind+1]+std_grouped[ind+1]], color="w", alpha=1)
    
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
        ax.set_axisbelow(True)
        ax.grid(axis='both', color='0.8')
        
    cax = fig.add_subplot(gs[0,1])
    cax.matshow(np.linspace(min_cv,max_cv,101).reshape(-1,1), cmap=cm, aspect=0.1)
    
    cax.set_xticks([])
    
    cax.yaxis.tick_right()
    cax.set_yticks(np.linspace(0,100,6), [str(round(float(label), 2)) for label in (np.linspace(0,1,6)*(max_cv-min_cv))+min_cv], fontsize=fs)
    
    cax.set_ylabel("Coefficient of variation", fontsize=fs, rotation=90)
    
    plt.savefig(outfile, bbox_inches="tight")
    
    if return_data:
        return (mean_grouped, std_grouped, cv_grouped)
        
# Enrichment    
def plot_enrichment(data_frame, keys, p_column, size_column, outfile = "../Out_enrichment.pdf", fs = 15, auc_column=None, cmap="Reds", splits=None, already_log10_transformed=False, v_lim=None, cbar_resolution = 4, num_legend_labels = 3, label=True, s_scale=1):

    if not already_log10_transformed:
        data_frame = data_frame.copy()
        data_frame[p_column] = np.log10(data_frame[p_column])
    
    if not v_lim:
        v_lim = (np.floor(np.nanmin(data_frame[p_column])),0)
    
    dv = v_lim[1]-v_lim[0]
    try:
        cm = mpl.cm.get_cmap(cmap)
    except:
        raise ValueError("Colormap not found")
    
    fig,ax=plt.subplots()
    
    if auc_column:
        fig.set_size_inches(1.5,len(keys)/2)
    else:
        fig.set_size_inches(0.5,len(keys)/2)
    
    sc = ax.scatter(np.ones(len(keys))*(-1),np.ones(len(keys))*(-1),s=data_frame.loc[list(keys.keys()),size_column],color="k", alpha=0.5)
    
    for ind,index in enumerate(keys.keys()):
        ax.scatter(1,ind,s=data_frame.loc[index,size_column]*s_scale,color=cm(-1*data_frame.loc[index,p_column]/dv))
        if auc_column:
            ax.text(1.02,ind,"{:.2f}".format(data_frame.loc[index,auc_column]),verticalalignment="center",fontsize=fs)
    
    if auc_column:
        ax.text(1.02,-1,"AUC", fontsize=fs, verticalalignment="center")
            
    ax.set_yticks(np.arange(0,len(keys)))
    ax.set_yticklabels([item[1] for item in keys.items()], fontsize=fs)
    
    ax.set_xticks([])
    ax.set_ylim(len(keys)-0.5,-0.5)
    if auc_column:
        ax.set_xlim(0.99,1.05)
    else:
        ax.set_xlim(0.99,1.01)
        
    if splits:
        for split in splits:
            ax.plot([0.99,1.08],[split+0.5,split+0.5],c="k",ls=":")
    
    if len(keys)<10:
        # Maybe add dynamic shrinkage?
        if auc_column:
            c_map_ax = fig.add_axes([1.0, 0.15, 0.1, 0.7])
        else:
            c_map_ax = fig.add_axes([1.3, 0.15, 0.45, 0.7])
    else:
        if auc_column:
            c_map_ax = fig.add_axes([1.0, 0.3, 0.1, 0.4])
        else:
            c_map_ax = fig.add_axes([1.3, 0.3, 0.45, 0.4])
    mpl.colorbar.ColorbarBase(c_map_ax, cmap=cmap, orientation = 'vertical')
    if label:
        c_map_ax.set_ylabel("log10(adj. p-value)",rotation=90, fontsize=fs, labelpad=10)
    
    c_map_ax.set_yticks(np.arange(v_lim[1],v_lim[0]-0.01,np.floor(v_lim[0]/cbar_resolution))/dv*(-1))
    c_map_ax.set_yticklabels(np.arange(v_lim[1],v_lim[0]-0.01,np.floor(v_lim[0]/cbar_resolution)),fontsize=fs)
    
    if len(keys)==1:
        ax.legend(*sc.legend_elements("sizes", num=num_legend_labels),fontsize=fs, ncol=1, bbox_to_anchor=(0.85, 0))
    else:
        ax.legend(*sc.legend_elements("sizes", num=num_legend_labels),fontsize=fs, ncol=2, bbox_to_anchor=(1.15, 0))
    plt.savefig(outfile,bbox_inches="tight")
       
def plot_AUC(data_frame_keys, keys, data_frame_ref_keys, ref_keys, outfile="Out_AUC.pdf", fs=15, grid=True, fs_legend=15):
            
    for ind,ref_key in enumerate(ref_keys):
        
        indices = [index for index in data_frame_ref_keys.index if index in data_frame_keys.index and ~np.isnan(data_frame_ref_keys.loc[index,ref_key])]
        
        fig,ax = plt.subplots()
        fig.set_size_inches(5,5)
        
        for key in keys:        
            fpr, tpr, _ = roc_curve(data_frame_keys.loc[indices,key],data_frame_ref_keys.loc[indices,ref_key])
            ax.plot(fpr, tpr, lw=5, label= keys[key]["Label"]+": {:.2f}".format(roc_auc_score(data_frame_keys.loc[indices,key],data_frame_ref_keys.loc[indices,ref_key])),c=keys[key]["Color"],zorder=20)
    
        # Plot reference line
        ax.plot([0,1],[0,1],ls="--",c="k",lw=3)
        
        # Set layout
        ax.set_xlabel("1-Specificity",fontsize=fs)
        ax.set_ylabel("Sensitivity",fontsize=fs)
        ax.set_xticks([0.25,0.5,0.75,1.0])
        ax.set_yticks([0.0,0.25,0.5,0.75,1.0])
        ax.set_xticklabels([0.25,0.5,0.75,1.0],fontsize=fs)
        ax.set_yticklabels([0.0,0.25,0.5,0.75,1.0],fontsize=fs)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        plt.legend(loc="lower right",fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
        
        if grid:
            ax.set_axisbelow(True)
            ax.grid(axis='both', color='0.8')
        
        if len(ref_keys)>1:
            plt.savefig(outfile[:-4]+"_"+str(ind)+outfile[-4:],bbox_inches="tight")  
        else:
            plt.savefig(outfile,bbox_inches="tight")  
            
def plot_EF(data_frame_keys, keys, data_frame_ref_keys, ref_keys, outfile="Out_EF.pdf", fs=15, grid=True, fs_legend=15, y_lim=None, legend_loc="upper right"):
            
    for ind,ref_key in enumerate(ref_keys):
        
        indices = [index for index in data_frame_ref_keys.index if index in data_frame_keys.index and ~np.isnan(data_frame_ref_keys.loc[index,ref_key])]
        
        y_min = 0
        y_max = 0
        
        fig,ax = plt.subplots()
        fig.set_size_inches(5,5)
        
        indices_sorted = data_frame_ref_keys.loc[indices,ref_key].sort_values(ascending=False).index
        rank = np.arange(1,len(indices_sorted)+1)
        
        for key in keys:        
                        
            enrichment = np.log10(np.cumsum(data_frame_keys.loc[indices_sorted,key].values)/(rank*np.sum(data_frame_keys.loc[indices_sorted,key].values)/np.max(rank)))
            
            filter_inf = np.isinf(enrichment)
            
            ax.plot(rank[~filter_inf],enrichment[~filter_inf], lw=3, label= keys[key]["Label"],c=keys[key]["Color"],zorder=20)
            
            y_min = np.min((y_min,np.min(enrichment[~filter_inf])))
            y_max = np.max((y_max,np.max(enrichment[~filter_inf])))
                
        if not y_lim:
            y_lim = (1.05*y_min, 1.05*y_max)
            
        # Plot reference line
        ax.plot([1,len(indices_sorted)],[0,0],ls="--",c="k",lw=3)
        
        # Set layout
        ax.set_xlabel("Rank",fontsize=fs)
        ax.set_ylabel("log10(Enrichment Factor)",fontsize=fs)
        ax.set_xticks([1,len(indices_sorted)])
        ax.tick_params(axis="both",labelsize=fs)
        ax.set_xlim(0,len(indices_sorted))
        ax.set_ylim(y_lim)
        
        try:
            plt.legend(loc=legend_loc,fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
        except:
            print("legend_loc not found. Using upper left as a default.")
            plt.legend(loc="upper right",fontsize=fs_legend, shadow=True, fancybox=True, framealpha=1)
        
        if grid:
            ax.set_axisbelow(True)
            ax.grid(axis='both', color='0.8')
        
        if len(ref_keys)>1:
            plt.savefig(outfile[:-4]+"_"+str(ind)+outfile[-4:],bbox_inches="tight")  
        else:
            plt.savefig(outfile,bbox_inches="tight")  
            
def plot_violin_upper_lower(data_frame, keys, ref_key, outfile="Out_violin.pdf", cut=0.25, fs=15, grid=True, split=False, y_lim=None):

    ref = list(ref_key.keys())[0]
    
    indices_hits = {ref:list(data_frame.index)}
    colors = [ref_key[ref]["Color"]]
    labels = ["all"]
    
    for key in keys:
        scores_sorted = data_frame[key].sort_values()
        indices_hits.update({key+"_flop":list(scores_sorted.index)[:int(np.floor(cut*len(scores_sorted)))]})
        colors.append(keys[key]["Color_lower"])
        labels.append("low")
        indices_hits.update({key+"_top":list(scores_sorted.index)[int(np.ceil((1-cut)*len(scores_sorted))):]})
        colors.append(keys[key]["Color_upper"])
        labels.append("high")
        
    data_to_plot = [data_frame.loc[indices_hits[key],ref].dropna().values for key in indices_hits]
    
    fig,ax = plt.subplots()
    fig.set_size_inches(1+1+(len(keys)*2),4+(len(keys)/2))
    bg = ax.violinplot(data_to_plot, showmedians=False, widths=0.75, showextrema=False)
    pl = ax.violinplot(data_to_plot, showmedians=True, widths=0.75, showextrema=True, quantiles=[[0.25,0.75]]*5)
    
    x_lim = (0.5,2*len(keys)+1.5)
    
    if split:
        dy = np.max(data_to_plot[0])-np.min(data_to_plot[0])
        for i in range(len(keys)):
            ax.plot([1.5+(i*2),1.5+(i*2)],[np.min(data_to_plot[0])-(0.05)*dy,np.max(data_to_plot[0])+(0.05)*dy],c="k",ls=":")
    
    if not y_lim:
        dy = np.max(data_to_plot[0])-np.min(data_to_plot[0])
        y_lim = (np.min(data_to_plot[0])-(0.05)*dy,np.max(data_to_plot[0])+(0.05)*dy)
    
    for pc in bg['bodies']:
        pc.set_facecolor("w")
        pc.set_edgecolor("w")
        pc.set_alpha(1)
    
    for pc, color in zip(pl['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        
    pl['cmedians'].set_color(colors)
    pl['cmedians'].set_linewidth(3)
    pl['cbars'].set_edgecolor(colors)
    pl['cmaxes'].set_alpha(0)
    pl['cmins'].set_alpha(0)
    pl['cquantiles'].set_color([el for item in [[color]*2 for color in colors] for el in item])
    pl['cquantiles'].set_linestyle(":")
    
    ax.set_xticks(list(range(1,(len(keys)*2+1)+1)))
    ax.set_xticklabels(labels,size=fs,rotation=45)
    ax.set_xlim(x_lim)
    
    ax.tick_params(axis="y",labelsize=fs)
    ax.set_ylabel(ref_key[ref]["Label"],fontsize=fs,labelpad=10)
    ax.set_ylim(y_lim)
    
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis='both', color='0.8')
        
    handles = [Patch([0],[0],color=keys[key]["Color"],label=keys[key]["Label"], alpha=0.5) for key in keys]
    legend = ax.legend(handles=handles, fontsize=fs, bbox_to_anchor=(0.5, -0.2), fancybox=True,shadow=True,loc="upper center")
    legend.get_frame().set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(outfile)
            
def plot_boxplot2_list(data, outfile="Out_boxplot2.pdf", labels=("1","2"), y_label="y", color_ref="C0", colors=("#1F87B4","#1F63B4"), do_stats=True, paired=False, grid=True, fs=15, fs_text=12, title=None, separate_ref_color=True):    

    if separate_ref_color:
        color_ref_0 = color_ref
        color_ref_1 = color_ref
    else:
        color_ref_0 = colors[0]
        color_ref_1 = colors[1]

    fig,ax = plt.subplots()
    fig.set_size_inches(2.5,5)
    
    data_wo_na = [[el for el in data[0] if not np.isnan(el)]]
    data_wo_na.append([el for el in data[1] if not np.isnan(el)])
    
    bplot = ax.boxplot(data_wo_na,patch_artist=True,medianprops=dict(color="k"),widths=0.8,showfliers=False)
    
    ax.scatter(np.random.uniform(0.65,1.35,len(data[0])),data[0],facecolor="w",edgecolor=color_ref_0,s=10)
    ax.scatter(np.random.uniform(1.65,2.35,len(data[1])),data[1],facecolor="w",edgecolor=color_ref_1,s=10)
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis='both', color='0.8')
    
    y_min = np.min((np.nanmin(data[0]),np.nanmin(data[1])))
    y_max = np.max((np.nanmax(data[0]),np.nanmax(data[1])))
    delta_y = y_max-y_min
    
    if do_stats:
        if paired:
            p_value = wilcoxon(data[0],data[1],nan_policy="omit")[1]
        else:
            p_value = ranksums(data[0],data[1],nan_policy="omit")[1]
        #print(p_value)
        ax.plot([1,1,2,2],[y_max+0.05*delta_y, y_max+0.1*delta_y, y_max+0.1*delta_y, y_max+0.05*delta_y],c="k",lw=1)
        if p_value<=1e-4:
            ax.text(1.5,y_max+0.12*delta_y,"***",horizontalalignment="center", fontsize=fs_text)
        elif p_value<=1e-3:
            ax.text(1.5,y_max+0.12*delta_y,"**",horizontalalignment="center", fontsize=fs_text)
        elif p_value<=1e-2:
            ax.text(1.5,y_max+0.12*delta_y,"*",horizontalalignment="center", fontsize=fs_text)
        else:
            ax.text(1.5,y_max+0.12*delta_y,"n.s.",horizontalalignment="center", fontsize=fs_text)
    
        y_lim = (y_min-0.05*delta_y,y_max+0.2*delta_y)
        
    else:
        y_lim = (y_min-0.05*delta_y,y_max+0.05*delta_y)
    
    ax.set_ylim(y_lim)
    ax.set_xticklabels(labels,fontsize=fs,rotation=90)
    ax.set_ylabel(y_label,fontsize=fs)
    ax.tick_params(axis="y",labelsize=fs)
    
    if title:
        ax.set_title(title, fontsize=fs)
    
    plt.tight_layout()
    plt.savefig(outfile)
    
def plot_bar(data, keys, outfile="Out_bar.pdf", method="mean", sort_values=False, fs=15, rotation=90, y_label="y", grid=True):
    
    keys_sorted = list(keys.keys())
    
    df_red = data[keys_sorted]
    
    if method not in ["mean","median"]:
        raise ValueError("method not found.")
    
    if method=="mean":
        values = df_red.mean()
        err = df_red.std()
    elif method=="median":
        values = df_red.median()
        err = (df_red - df_red.median()).abs().median()
    
    if sort_values:
        keys_sorted = list(values.sort_values(ascending=False).index)
    
    labels = [keys[key]["Label"] for key in keys_sorted]
    colors = [keys[key]["Color"] for key in keys_sorted]
    
    fig,ax = plt.subplots()
    fig.set_size_inches(6,4)
    try:
        ax.bar(np.arange(0,len(keys)),values[keys_sorted], yerr=err[keys_sorted], color=colors, edgecolor="k")
    except:
        ax.bar(np.arange(0,len(keys)),values[keys_sorted], yerr=err[keys_sorted], edgecolor="k")
    ax.set_xticks(np.arange(0,len(keys)))
    ax.set_xticklabels(labels, fontsize=fs, rotation = rotation)
    ax.tick_params(axis="y",labelsize=fs)
    ax.set_ylabel(y_label, fontsize=fs)
    ax.set_xlim(-1,len(keys))
    
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis='both', color='0.8')    
        
    plt.tight_layout()
    plt.savefig(outfile)

def plot_violin_quantiles(data,outfile="Out_violin2_list.png", labels=("1","2"), y_label="y", colors=("C0","C1"), grid=True, fs=15, rotation=90, y_lim = None, lines = None, fs_text=15,title=None, fs_title=15, calc_widths=False):
    
    ### For width add density normailzation
    
    if calc_widths:
        flat = [el for item in data for el in item]
        widths = [np.max(np.histogram(data[ind],bins=10,range=(np.min(flat),np.max(flat)))[0]) for ind in range(len(data))]
        widths = widths/max(widths)*0.75
    else:
        widths = 0.75
    
    fig,ax = plt.subplots()
    fig.set_size_inches(4,4)
    bg = ax.violinplot(data, showmedians=False, widths=widths, showextrema=False)
    pl = ax.violinplot(data, showmedians=True, widths=widths, showextrema=True, quantiles=[[0.25,0.75]]*len(labels))
    
    for pc in bg['bodies']:
        pc.set_facecolor("w")
        pc.set_edgecolor("w")
        pc.set_alpha(1)
    
    for pc, color in zip(pl['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        
    pl['cmedians'].set_color(colors)
    pl['cmedians'].set_linewidth(3)
    pl['cbars'].set_edgecolor(colors)
    pl['cmaxes'].set_alpha(0)
    pl['cmins'].set_alpha(0)
    pl['cquantiles'].set_color([el for item in [[color]*2 for color in colors] for el in item])
    pl['cquantiles'].set_linestyle(":")
    
    if title:
        ax.set_title(title,fontsize=fs_title)
    
    ax.set_xticks(np.arange(1,len(labels)+1,1))
    ax.set_xticklabels(labels,size=fs,rotation=rotation, ha='right', va='top')
    
    ax.tick_params(axis="y",labelsize=fs)
    ax.set_ylabel(y_label,fontsize=fs,labelpad=10)
    
    if y_lim:
        ax.set_ylim(y_lim)          
    
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis='both', color='0.8')
    
    if lines:
        for el in lines:
            ax.plot([el[0],el[1]],[el[2],el[2]],c="k")
            ax.text((el[0]+el[1])/2,el[2],el[3],horizontalalignment="center", fontsize=fs_text)
    
    plt.tight_layout()
    plt.savefig(outfile)
    
def plot_violin_grid(data_dict, keys, style_dict, legend_dict=None, outfile="Out_violin2_grid.png", labels=("1","2"), y_label="y", colors=None, grid=True, fs=15, rotation=90, y_lim = None, lines = None, fs_text=15,title=None, fs_title=15, n_columns=2, dy=0.1, calc_widths=False):
    
    n_rows = int(np.ceil(len(data_dict)/n_columns))
            
    fig = plt.figure()
    fig.set_size_inches((n_columns*3)+1.2,(n_rows*3)+0.9)
    gs = gridspec.GridSpec(n_rows+1, n_columns+1, wspace=0.4, hspace=0.4, width_ratios=[1]+[10]*n_columns, height_ratios=[10]*n_rows+[3])
    
    for key in data_dict:
        
        data = [data_dict[key][key_y][~np.isnan(data_dict[key][key_y])] for key_y in keys]        
        title = style_dict[key]["Label"]
        colors = [style_dict[key]["Color"]]*len(data)
        ind = style_dict[key]["Order"]
        
        if calc_widths:
            flat = [el for item in data for el in item]
            widths = [np.max(np.histogram(data[ind],bins=10, range=(np.min(flat),np.max(flat)))[0]) for ind in range(len(data))]
            widths = widths/max(widths)*0.75
        else:
            widths = 0.75
    
        ax = fig.add_subplot(gs[int(np.floor(ind/n_columns)),int(np.mod(ind,n_columns))+1])  
        
        bg = ax.violinplot(data, showmedians=False, widths=widths, showextrema=False)
        pl = ax.violinplot(data, showmedians=True, widths=widths, showextrema=True, quantiles=[[0.25,0.75]]*len(labels))
        
        for pc in bg['bodies']:
            pc.set_facecolor("w")
            pc.set_edgecolor("w")
            pc.set_alpha(1)
        
        for pc, color in zip(pl['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            
        pl['cmedians'].set_color(colors)
        pl['cmedians'].set_linewidth(3)
        pl['cbars'].set_edgecolor(colors)
        pl['cmaxes'].set_alpha(0)
        pl['cmins'].set_alpha(0)
        pl['cquantiles'].set_color([el for item in [[color]*2 for color in colors] for el in item])
        pl['cquantiles'].set_linestyle(":")
        
        if title:
            ax.set_title(title,fontsize=fs_title)
        
        ax.set_xticks(np.arange(1,len(labels)+1,1))
        if int(np.floor(ind/n_columns)) == n_rows-1:
            ax.set_xticklabels(labels,size=fs,rotation=rotation, ha='center', va='top')
        else: 
            ax.set_xticklabels([])
        
        ax.tick_params(axis="y",labelsize=fs)
        
        if y_lim:
            ax.set_ylim(y_lim)          
        
        if grid:
            ax.set_axisbelow(True)
            ax.grid(axis='both', color='0.8')
        
    fig.text(dy, 0.5, y_label, rotation="vertical", va="center", fontsize=fs)
    
    if legend_dict:
        legend_elements = [Patch(facecolor=legend_dict[key]["Color"],label=legend_dict[key]["Label"]) for key in legend_dict if key != "Columns"]
        fig.legend(loc="lower center", handles=legend_elements, ncol=legend_dict["Columns"],fontsize=fs)

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")    

def plot_completeness(data,label_dict,outfile="Completeness.pdf",cmap="binary",fs=15,color_counts_y = "k", color_counts_x = "k", highlight_quartiles=True, color_highlight="#808B96", xlabel = "x", sort_values = True):

    if sort_values:
        data = data[data.sum().sort_values().index[::-1]]
    
    fig = plt.figure()
    fig.set_size_inches(8,8)
    gs = gridspec.GridSpec(2,2,width_ratios=[10,2],height_ratios=[2,10], wspace=0.05, hspace=0.05)
    
    ax_comp = fig.add_subplot(gs[1,0])
    ax_comp.matshow(data.values,aspect="auto",cmap=cmap)
    
    #ax_comp.set_xticks([0,len(data.T)-1])
    ax_comp.set_xticks([])
    ax_comp.set_xlabel(xlabel,fontsize=fs)
    ax_comp.tick_params(axis="x",labelsize=fs)
    ax_comp.set_yticks(range(0,len(data)))
    ax_comp.set_yticklabels([label_dict[index] for index in list(data.index)],fontsize=fs)
    ax_comp.xaxis.tick_bottom()
    
    ax_counts_y = fig.add_subplot(gs[1,1])
    ax_counts_y.barh(range(0,len(data)),data.T.sum().values,height=1,color=color_counts_y,align="edge")
    ax_counts_y.set_xlabel("Counts",fontsize=fs)
    ax_counts_y.set_xticks([0,len(data.T)-1])
    ax_counts_y.tick_params(axis="x",labelsize=fs)
    ax_counts_y.set_yticks([])
    
    ax_counts_y.set_ylim([len(data),0])
    ax_counts_y.set_xlim([0,len(data.T)-1])
    
    ax_counts_x = fig.add_subplot(gs[0,0])
    ax_counts_x.bar(range(0,len(data.T)),data.sum().values,width=1,color=color_counts_x)
    ax_counts_x.set_yticks([1,len(data)])
    ax_counts_x.set_ylabel("Counts",fontsize=fs)
    ax_counts_x.tick_params(axis="y",labelsize=fs)
    ax_counts_x.set_xticks([])
    
    ax_counts_x.set_xlim([0,len(data.T)-1])
    ax_counts_x.set_ylim([0,len(data)])
    
    if highlight_quartiles:
        for split in [.25,.5,.75]:
            ax_counts_y.plot([split*len(data.T),split*len(data.T)],[0,len(data)],ls=":",c=color_highlight,lw=3)
            ax_counts_x.plot([split*len(data.T),split*len(data.T)],[0,len(data)],ls=":",c=color_highlight,lw=3)
    
    plt.savefig(outfile,bbox_inches="tight")
