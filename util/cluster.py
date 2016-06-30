#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import print_function

from herdingspikes import *
import os
from datetime import datetime
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")

def spike_sort(path,spikefile,
    shoffset = 0,
    upto = 0,
    align = False,
    noShapes = False,
    presortSpikes = True,
    alpha = 0.3,
    h = 0.3,
    ampthreshold = 5):
    '''
    Matthia's Hennig's spike sorting script. 
    Converted by M. Rule 2016 06 30
    
    Parameters
    ----------
    path : string
        Path to folder containing raw spiking data. Should end in path
        separator.
    spikefile : string
        File name to spike sort
    shoffset : int
        Undocumented
    upto : int
        This is used to select only a part of the spike shapes for 
        feature extraction. Has to be used for recordings at higher
        sampling rate (18,24kHz). Leave at zero if not needed
    align : bool
        set to True if spike shapes should be re-aligned, should 
        generally not be required, but can improve clustering a bit
        in some cases
    noShapes : bool
        set to True if shapes should not be saved 
        (gives much smaller file)
    presortSpikes : bool
        set to True if spikes should be pre-filtered, advisable at 
        low samplig rates
    alpha : float
        relative weight of PCA vs. locations in clustering,
        default of 0.3 should be ok
    h : float
        kernel size for clustering, default of 0.3 should be ok
    ampthreshold : int
        for presorting, spikes with larger amplitude will be used to
        train the classifier for true spikes. Default 5
    '''

    # load spike data
    outfile = path+spikefile+'_clustered_'+str(h)+'_'+str(alpha)
    if align: outfile += '_align

    if presortSpikes:
        outfile = outfile + "_presorted2"
    if os.path.isfile(outfile + '.hdf5'):
        print("Deleting old file:" + outfile + '.hdf5')
        os.remove(outfile + '.hdf5')
    O = ImportInterpolated(path + spikefile + '.hdf5')

    # align shapes
    if align:
        print("Aligning shapes...",)
        O.AlignShapes()
        print("done.")

    # presort, remove noise
    if presortSpikes:
        print("Selecting good spikes...",)
        scorePCA = O.ShapePCA(
            ncomp=6, white=True, offset=shoffset, upto=upto)
        nbins = [64, 64]

        l = O.Locations()
        hg, bx, by = np.histogram2d(l[0], l[1], nbins)

        mindensity = np.min(hg[hg > 0])
        densitythreshold = np.max(
            (np.percentile(hg.flatten(), 0.5), mindensity + 1))
        print("Minimum density: ", mindensity)
        print("Density threshold: ", densitythreshold)
        print("Amplitude threshold: ", ampthreshold)
        nGood = 2000
        nBad  = 2000

        binspanx = (np.max(l[0])-np.min(l[0]))/nbins[0]*1.001
        binspany = (np.max(l[1])-np.min(l[1]))/nbins[1]*1.001
        nbx = ((l[0]-np.min(l[0]))//binspanx).astype(int)
        nby = ((l[1]-np.min(l[1]))//binspany).astype(int)
        indbad = np.where(hg[nbx, nby] <= densitythreshold)[0]
        indbad = np.random.permutation(indbad)[:nBad]
        print("Classifier is based on " +
              str(len(indbad)) + " examples of bad shapes ",)
        badshape = np.median(O.Shapes()[:, indbad], axis=1)

        fakeampl = -np.min(O.Shapes(), axis=0)
        indgood = np.where(fakeampl > ampthreshold)[0]
        indgood = np.random.permutation(indgood)[:nGood]
        print("and " + str(len(indgood)) + " examples of good shapes.")
        goodshape = np.median((O.Shapes()[:, indgood]), axis=1)

        print('Spikes in both:', np.intersect1d(indbad, indgood))

        labels = np.append(np.zeros(len(indbad)), np.ones(len(indgood)))
        pcs = np.hstack((scorePCA[:, indbad], scorePCA[:, indgood]))
        classifier = svm.SVC(kernel='rbf', class_weight='balanced')
        classifier.fit(pcs.T, labels)
        score = classifier.predict(scorePCA.T).astype(int)
        print("bad:", np.sum(score == 0), "good:", np.sum(score == 1))

        fig = plt.figure(figsize=(10, 4 * 3))
        nbins = [64 * 2, 64 * 2]
        cmap = plt.cm.RdBu_r
        cmap = plt.cm.gist_earth
        cmap.set_bad('k')

        ax = plt.subplot(321)
        ax.set_axis_bgcolor('black')
        hg, bx, by = np.histogram2d(l[0], l[1], nbins)
        rateMasked = np.ma.array(hg, mask=(hg == 0))
        plt.pcolor(bx, by, np.log10(rateMasked), cmap=cmap, vmin=0, vmax=3)
        plt.colorbar()
        plt.axis('equal')
        plt.xlim((0, 65))
        plt.ylim((0, 65))
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.title('all spikes')

        ax = plt.subplot(322)
        ax.set_axis_bgcolor('black')
        inds = np.where(score == 1)[0]
        hg1, bx, by = np.histogram2d(l[0][inds], l[1][inds], nbins)
        rateMasked = np.ma.array(hg1, mask=(hg1 == 0))
        plt.pcolor(bx, by, np.log10(rateMasked), cmap=cmap, vmin=0, vmax=3)
        plt.colorbar()
        plt.axis('equal')
        plt.xlim((0, 65))
        plt.ylim((0, 65))
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.title('good spikes')

        ax = plt.subplot(324)
        ax.set_axis_bgcolor('black')
        inds = np.where(score == 0)[0]
        hg12, bx, by = np.histogram2d(l[0][inds], l[1][inds], nbins)
        rateMasked = np.ma.array(hg12, mask=(hg12 == 0))
        plt.pcolor(bx,by,np.log10(rateMasked),cmap=cmap,vmin=0,vmax=3)
        plt.colorbar()
        plt.axis('equal')
        plt.xlim((0, 65))
        plt.ylim((0, 65))
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.title('bad spikes')

        ax = plt.subplot(326)
        ax.set_axis_bgcolor('black')
        rateMasked = np.ma.array(hg12 / hg, mask=(hg == 0))
        plt.pcolor(bx,by,rateMasked,cmap=cmap,vmin=0,vmax=1)
        plt.colorbar()
        plt.axis('equal')
        plt.xlim((0, 65))
        plt.ylim((0, 65))
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.title('fraction bad')

        ax = plt.subplot(325)
        plt.plot(badshape, label='bad')
        plt.plot(goodshape, label='good')
        plt.legend()

        ax = plt.subplot(323)
        nShow = 7000
        inds = np.where(score == 0)[0]
        plt.scatter(scorePCA[0, inds[:nShow]], scorePCA[
                    1, inds[:nShow]], c='r', s=6)
        inds = np.where(score == 1)[0]
        plt.scatter(scorePCA[0, inds[:nShow]], scorePCA[
                    1, inds[:nShow]], c='b', s=6)

        fname = path + spikefile + '_density.png'
        plt.savefig(fname)
        plt.close()

        O.KeepOnly(np.where(score == 1)[0])
        print("done")

    # cluster data
    scorePCA = O.ShapePCA(ncomp=2, white=True, offset=shoffset, upto=upto)
    startTime = datetime.now()
    O.CombinedMeanShift(h, alpha, scorePCA, mbf=5)
    print('Time taken for sorting: ' + str(datetime.now() - startTime))

    # save data
    startTime = datetime.now()
    print(outfile)
    if not noShapes:
        O.Save(outfile + '.hdf5', 'lzf')
    else:
        g = h5py.File(outfile + '_noshapes.hdf5', 'w')
        g.create_dataset("data", data=O.Locations())
        g.create_dataset("centres", data=O.ClusterLoc())
        g.create_dataset("cluster_id", data=O.ClusterID())
        g.create_dataset("times", data=O.Times())
        g.create_dataset("Sampling", data=O.Sampling())
        g.close()

    print('Time for saving: ' + str(datetime.now() - startTime))
