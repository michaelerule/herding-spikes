import numpy as np
import h5py.h5ac
#import scipy.special._ufuncs_cxx


class DataBase():

    def setupDatabase(self, spikeFile):
        f=h5py.File(spikeFile,'r')
        self.clusterID = np.array(f['cluster_id'].value,dtype=int)
        self.shapes = np.array(f['shapes'].value,dtype=float)
        self.data = np.array(f['data'].value,dtype=float)
        self.centres = np.array(f['centres'].value,dtype=float)
        self.times = np.array(f['times'].value,dtype=int)
        self.sampling = f['Sampling'].value
        f.close()

    def getClusterID(self):
        return self.clusterID

    def getShapes(self):
        return self.shapes

    def getData(self):
        return self.data

    def getCentres(self):
        return self.centres

    def getTimes(self):
        return self.times

    def getSampling(self):
        return self.sampling
