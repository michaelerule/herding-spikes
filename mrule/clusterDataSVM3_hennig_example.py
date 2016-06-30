from herdingspikes import *
import os
from datetime import datetime
from sklearn import svm

if __name__ == '__main__':

    # file(s) to sort

    # example how to sort multiple files in one go (together)
    path = '/disk/scratch/mhennig/P29_16_07_14/HdfFilesSpkD45_dev/'
    sfs = (
        'P29_16_05_14_retina02_left_spntdark_v28.hdf5',
        'P29_16_05_14_retina02_left_stim1_classtimnew_v28.hdf5',
        'P29_16_05_14_retina02_left_stim2_smallarray_fullfield_v28.hdf5'
        )

    for i in range(len(sfs)):
        sfs[i] = path + sfs[i]
    multiFile = True

    # or here a single file (comment out for multiple files)
    path = '/disk/scratch/mhennig/P29_16_07_14/HdfFilesSpkD45_dev/'
    spikefile1 = 'P29_16_05_14_retina02_left_stim1_classtimnew_v28'
    multiFile = False

    # Parameters

    # This is used to select only a part of the spike shapes for feature
    # extraction. Has to be used for recordings at higher sampling rate
    # (18,24kHz). Leave at zero if not needed
    shoffset = 0
    upto = 0

    # set to True if spike shapes should be re-aligned, should generally not
    # be required, but can improve clustering a bit in some cases
    align = False

    # set to True if shapes should not be saved (gives much smaller file)
    noShapes = False  # True

    # set to True if spikes should be pre-filtered, advisable at low samplig
    # rates
    presortSpikes = True

    # relative weight of PCA vs. locations in clustering, default should be ok
    alpha = 0.3

    # kernel size for clustering, default should be ok
    h = 0.3

    # for presorting, spikes with larger amplitude will be used to train the classifier
    # for true spikes
    ampthreshold = 5

    # load spike data
    if not multiFile:
        if align:
            outfile = path + spikefile1 + '_clustered_' + \
                str(h) + '_' + str(alpha) + '_align'
        else:
            outfile = path + spikefile1 + \
                '_clustered_' + str(h) + '_' + str(alpha)

        if presortSpikes:
            outfile = outfile + "_presorted2"
        if os.path.isfile(outfile + '.hdf5'):
            print("Deleting old file:" + outfile + '.hdf5')
            os.remove(outfile + '.hdf5')
        O = ImportInterpolated(path + spikefile1 + '.hdf5')
    else:
        O = ImportInterpolatedList(sfs)
        if align:
            outpostfix = '_clustered_' + \
                str(h) + '_' + str(alpha) + '_align_multi'
        else:
            outpostfix = '_clustered_' + str(h) + '_' + str(alpha) + '_multi'
        if presortSpikes:
            outpostfix = outpostfix + "_presorted2"

    # align shapes
    if align:
        print("Aligning shapes...",)
        O.AlignShapes()
        print("done.")

    # presort, remove noise
    if presortSpikes:
        print("Selecting good spikes...",)
        scorePCA = O.ShapePCA(ncomp=6, white=True, offset=shoffset, upto=upto)
        nbins = [64, 64]

        l = O.Locations()
        hg, bx, by = np.histogram2d(l[0], l[1], nbins)

        mindensity = np.min(hg[hg > 0])
        densitythreshold = np.max(
            (np.percentile(hg.flatten(), 0.5), mindensity + 1))
        print("Minimum density: ", mindensity)
        print("Density threshold: ", densitythreshold)
        print("Amplitude threshold: ", ampthreshold)
        nGood = 2000  # O.NData()/40 #20000 (40)
        nBad = 2000  # O.NData()/1000 #1000 (800)

        binspanx = (np.max(l[0]) - np.min(l[0])) / nbins[0] * 1.001
        binspany = (np.max(l[1]) - np.min(l[1])) / nbins[1] * 1.001
        nbx = ((l[0] - np.min(l[0])) // binspanx).astype(int)
        nby = ((l[1] - np.min(l[1])) // binspany).astype(int)
        indbad = np.where(hg[nbx, nby] <= densitythreshold)[0]
        indbad = np.random.permutation(indbad)[:nBad]
        print("Classifier is based on " +
              str(len(indbad)) + " examples of bad shapes ",)
        badshape = np.median(O.Shapes()[:, indbad], axis=1)

        fakeampl = -np.min(O.Shapes(), axis=0)
        indgood  = np.where(fakeampl > ampthreshold)[0]
        indgood  = np.random.permutation(indgood)[:nGood]
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
        plt.pcolor(bx, by, np.log10(rateMasked), cmap=cmap, vmin=0, vmax=3)
        plt.colorbar()
        plt.axis('equal')
        plt.xlim((0, 65))
        plt.ylim((0, 65))
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.title('bad spikes')

        ax = plt.subplot(326)
        ax.set_axis_bgcolor('black')
        rateMasked = np.ma.array(hg12 / hg, mask=(hg == 0))
        plt.pcolor(bx, by, rateMasked, cmap=cmap, vmin=0, vmax=1)
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

        if not multiFile:
            fname = path + spikefile1 + '_density.png'
        else:
            fname = sfs[0].replace('.hdf5', '') + '_density.png'
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
    if not multiFile:
        print(outfile)
        if not noShapes:
            O.Save(outfile + '.hdf5', 'lzf')
        else:
            g = h5py.File(outfile + '_noshapes.hdf5', 'w')
            g.create_dataset("data", data=O.Locations())
            g.create_dataset("centres", data=O.ClusterLoc())
            g.create_dataset("cluster_id", data=O.ClusterID())
            g.create_dataset("times", data=O.Times())
            # g.create_dataset("shapes",data=O.Shapes()[:,inds],compression='lzf')
            g.create_dataset("Sampling", data=O.Sampling())
            g.close()

    else:
        for c, name in enumerate(sfs):
            if not noShapes:
                ofname = name.replace('.hdf5', '') + outpostfix + '.hdf5'
            else:
                ofname = name.replace('.hdf5', '') + \
                    outpostfix + '_noshapes.hdf5'
            print(ofname)
            inds = O.ExperimentIndices(c)
            g = h5py.File(ofname, 'w')
            g.create_dataset("data", data=O.Locations()[:, inds])
            # g.create_dataset("expinds",data=self.__expinds)
            g.create_dataset("centres", data=O.ClusterLoc())
            g.create_dataset("cluster_id", data=O.ClusterID()[inds])
            g.create_dataset("times", data=O.Times()[inds])
            if not noShapes:
                g.create_dataset("shapes", data=O.Shapes()[
                                 :, inds], compression='lzf')
            g.create_dataset("Sampling", data=O.Sampling())
            g.close()

    print('Time for saving: ' + str(datetime.now() - startTime))
