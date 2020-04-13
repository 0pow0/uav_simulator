import numpy as np
# from simulator import Simulator
from simulatorNFZ import SimulatorNFZ
# from simulator_astar import SimulatorAstar
# from simulator_astarOnes import SimulatorAstarOnes
# from simulator_astarProb import SimulatorAstarProb
import logging
import time
import os

np.random.seed(0)


# from simulator import Simulator
# from simulator_randTask import Simulator
# from simulator_routing import Simulator


class Preprocess:

    def __init__(self, label=None, mainList=None, subOutput=None, subOutputCube=None, rfeature=None, subList=None,
                 subLabel=None, noFlyZone=None):
        logging.info('')
        logging.info('---Initial Shape---')
        print('\n---Initial Shape---')
        if label is None:
            print("MainNet label is none")
        else:
            self.gtr = label
            logging.info('  initial MainNet label: {0}\n'.format(self.gtr.shape))
            print('initial MainNet label: ', self.gtr.shape)

        if mainList is None:
            print("MainNet tasklist is none")
        else:
            self.mt = mainList
            logging.info('  initial MainNet tasklist: {0}'.format(self.mt.shape))
            print('initial MainNet tasklist', self.mt.shape)

        if subOutput is None:
            print("subNet Output is none")
        else:
            self.subOutput = subOutput
            logging.info('  initial subNet Output: {0}'.format(self.subOutput.shape))
            print('initial subNet Output', self.subOutput.shape)

        if subOutputCube is None:
            print("subNet Output Cube is none")
        else:
            self.subOutputCube = subOutputCube
            logging.info('  initial subNet Cube Output: {0}'.format(self.subOutputCube.shape))
            print('initial subNet Cube Output', self.subOutputCube.shape)

        if rfeature is None:
            print("rnet feature is none")
        else:
            self.rfeature = rfeature
            logging.info('  initial rnet feature: {0}'.format(self.rfeature.shape))
            print('initial rnet feature', self.rfeature.shape)

        if subList is None:
            print("subNet tasklist is none")
        else:
            self.st = subList
            logging.info('  initial subNet tasklist: {0}\n'.format(self.st.shape))
            print('initial subNet tasklist: {0}'.format(self.st.shape))

        if subLabel is None:
            print("subLabel is none")
        else:
            self.sl = subLabel
            logging.info('  initial subLabel: {0}\n'.format(self.sl.shape))
            print('initial subLabel: {0}\n'.format(self.sl.shape))

        if noFlyZone is None:
            print("noFlyZone is none")
        else:
            self.nfz = noFlyZone
            logging.info('  noFlyZone: {0}\n'.format(self.nfz.shape))
            print('noFlyZone: {0}\n'.format(self.nfz.shape))

        print('')

    # save data from start to end
    def splitByTime(self, start=0, end=0):
        if end == 0:
            self.gtr = self.gtr[:, start:]
            # self.tsr = self.tsr[:, start:]
        else:
            self.gtr = self.gtr[:, start:end]
            # self.tsr = self.tsr[:, start:end]
        logging.info(self.tsr.shape)
        logging.info(self.gtr.shape)
        logging.info('splitByTime complete\n')

    # switch all elements to zero or one 
    def oneOrZero(self, gtr):
        m = np.median(gtr[gtr != 0])
        logging.info('median: {0}'.format(m))
        # self.gtr[self.gtr<=m] = 0
        # self.gtr[self.gtr>m] = 1
        gtr[gtr < m] = 0
        gtr[gtr >= m] = 1
        logging.info('oneOrZero complete\n')
        return gtr

    def densityToOne(self, gtr):
        gtr[gtr > 0] = 1
        logging.info('densityToOne complete\n')
        return gtr

    # ground truth only save the last second (the 30th second)
    def lastSecond(self):
        gtr1 = self.gtr[:, 29:, :, :].reshape((1, 16, 16))
        print('self.gtr[:,29:,:,:]: ', self.gtr[:, 29:, :, :].shape)
        print('gtr1: ', gtr1.shape)
        print('self.gtr == gtr1:', np.all(gtr1 == self.gtr[:, 29]))
        self.gtr = gtr1
        print('lastSecond complete\n')

    def checkDataIdentical(self, data1, data2):
        # p = np.random.randint(0, len(data1), 5)
        for i in range(0, 5):
            logging.info(np.all(data1[i] == data2[i]))
        logging.info('check complete\n')

    def compressTime(self):
        # feature: (n, 200, 10,  5) --> (n, 20, 100, 5)
        # label  : (n, 200, 100, 100) --> (n, 20, 100, 100)
        # nf : (n, 20, 100, 5)
        # nl : (n, 20, 100, 100)

        nf = np.zeros((self.tsr.shape[0], int(self.tsr.shape[1] / 10), self.tsr.shape[2] * 10, self.tsr.shape[3]))
        nl = np.zeros((self.gtr.shape[0], int(self.gtr.shape[1] / 10), self.gtr.shape[2], self.gtr.shape[3]))

        for i in range(10):
            ft, lb = self.tsr[i], self.gtr[i]
            for it in range(10, 201, 10):
                time_idx = int(it / 10) - 1
                # every sample, generate density map in 10 time intervel
                nl[i, time_idx] = np.sum(lb[it - 10:it], axis=0) / 10
                # every sample, record all task
                task_num = 0
                for j in range(it - 10, it):
                    for k in range(10):
                        nf[i, time_idx, task_num, :] = ft[i, j, k, :]
                        task_num += 1
            nl[i] = self.batchNormalize(nl[i])
        self.tsr = nf
        self.gtr = nl

    def standardDeviation(self, interval=15):
        # lb: (n, 60, 100, 100) --> (n, 100, 100)
        # nf: (n, 60, 100, 100) --> (n, 30, 100, 100)
        batchNum, intervalNum, row, col = self.gtr.shape
        intervalNum -= 2 * interval
        lb = np.zeros((batchNum, row, col))
        nf = np.zeros((batchNum, intervalNum, row, col))
        for b in range(batchNum):
            lb[b] = np.sum(self.gtr[b, intervalNum + interval:intervalNum + interval * 2], axis=0)
            for i in range(intervalNum):
                print(i, i + interval)
                nf[b, i] = np.sum(self.gtr[b, i:i + interval], axis=0)

    # print number of non-zeros and zeros
    def computeWeights(self, gtr):
        one = gtr[gtr > 0].size
        zero = gtr[gtr == 0].size
        logging.info('zero: {0}'.format(zero))
        logging.info('one: {0}'.format(one))
        logging.info('weight: {0}'.format(zero / one))
        logging.info('computeWeights complete\n')

    # nomalize groud truth as the last second
    def batchNormalize(self, gtr):
        # for i in range(len(gtr)):
        #     gtr[i] = (gtr[i] - np.min(gtr[i])) / (np.max(gtr[i]) - np.min(gtr[i]))
        if np.max(gtr) != 0:
            gtr = (gtr - np.min(gtr)) / (np.max(gtr) - np.min(gtr))
        # logging.info('      after batchNormalize')
        # print('          min: {0}'.format(np.min(gtr)))
        # print('          max: {0}'.format(np.max(gtr)))
        # print('          mean: {0}'.format(np.mean(gtr)))
        # print('          median: {0}'.format(np.median(gtr)))
        return gtr

    # lumped map divided time, return with batch normalization
    def averageDensity(self, gtr, time):
        gtr = gtr / time
        print(gtr.shape)
        return gtr

    # broadcast one sample to many
    def broadCast(self):
        self.tsr = np.broadcast_to(self.tsr, (10000, 30, 32, 32, 4))
        self.gtr = np.broadcast_to(self.gtr, (10000, 30, 32, 32))
        print(self.tsr.shape)
        print(self.gtr.shape)
        print('broadCast complete\n')

    # (30, 32, 32) --> (32, 32)
    def generateDensity(self, gtr):
        temp = np.sum(gtr, axis=1)
        logging.info('      density map is {0}'.format(temp.shape))
        return temp

    def generatePattern(self, gtr):
        temp = np.sum(gtr, axis=1)
        temp[temp > 0] = 1
        logging.info(temp.shape)
        logging.info('generatePattern complete\n')
        return temp

    def save(self, data, name='test', directory='test', subDirectory='subtest'):
        if not os.path.exists('/home/share_uav/ruiz/data/{0}'.format(directory)):
            os.mkdir('/home/share_uav/ruiz/data/{0}'.format(directory))
            os.chmod('/home/share_uav/ruiz/data/{0}'.format(directory), 0o777)
        if not os.path.exists('/home/share_uav/ruiz/data/{0}/{1}'.format(directory, subDirectory)):
            os.mkdir('/home/share_uav/ruiz/data/{0}/{1}'.format(directory, subDirectory))
            os.chmod('/home/share_uav/ruiz/data/{0}/{1}'.format(directory, subDirectory), 0o777)

        if os.path.exists(
                '/home/share_uav/ruiz/data/{0}/{1}/{2}.npy'.format(directory, subDirectory, name)):
            os.remove('/home/share_uav/ruiz/data/{0}/{1}/{2}.npy'.format(directory, subDirectory, name))

        np.save('/home/share_uav/ruiz/data/{0}/{1}/{2}.npy'.format(directory, subDirectory, name), data)
        os.chmod('/home/share_uav/ruiz/data/{0}/{1}/{2}.npy'.format(directory, subDirectory, name),
                 0o777)
        print(' {0}/{1}: {2} save complete\n'.format(subDirectory, name, data.shape))

    # generate density map from timestep start -> end
    # 每一个长度为70的时间段内，
    # (600, 10, 100, 100)
    def intervalDensity(self, data, start, end):
        logging.info('      generate density map from {0} to {1}'.format(start, end))
        # (600, 10, 100, 100)
        interval = data[:, start:end]
        # (600, 1, 100, 100)
        densityMap = self.generateDensity(interval)
        return self.averageDensity(densityMap, end - start)

    def trajectoryDivdeMerge(self, matri, start, end, segtime=60):
        n = int(matri.shape[1] / segtime)
        logging.info(
            '      split matrix into {0} parts and generate density map for each from {1} to {2}, then merge'.format(n,
                                                                                                                     start,
                                                                                                                     end))
        tempMtr = np.zeros(shape=(matri.shape[0], n, matri.shape[2], matri.shape[3]))
        for i in range(n):
            # (600, 70, 100, 100)
            tempMtrA = matri[:, i * 60:i * 60 + 70]
            tempMtrA = self.intervalDensity(tempMtrA, start, end)
            tempMtr[:, i] = tempMtrA
        return tempMtr

    def taskTimeDivde(self, matrix, segtime=60):
        n = int(matrix.shape[1] / segtime)
        tempMtr = np.zeros(shape=(matrix.shape[0], n, segtime, matrix.shape[2], matrix.shape[3]))
        for i in range(n):
            tempMtr[:, i] = matrix[:, i * 60:(i + 1) * 60]
        return tempMtr

    def fieldTransformer(self, data: np.ndarray, areaSize):
        mapSize = data.shape[-1]
        tmpSize = list(data.shape)
        # Index : (100,100) => (98,98) [0,97]
        sizeAfterTransform = mapSize - areaSize + 1
        tmpSize[-1] = sizeAfterTransform
        tmpSize[-2] = sizeAfterTransform
        tmpSize[-3] = 1
        tmpMin = np.zeros(shape=tmpSize)
        tmpMax = np.zeros(shape=tmpSize)

        for i in range(sizeAfterTransform):
            for j in range(sizeAfterTransform):
                tmp1 = np.sum(data[..., i:i + 3, j:j + 3], axis=(-1, -2))

                tmpMin[..., i, j] = np.min(tmp1, axis=-1, keepdims=True)
                tmpMax[..., i, j] = np.max(tmp1, axis=-1, keepdims=True)

        return tmpMin, tmpMax

    def featureMinMax(self, directory="minmax"):
        logging.info('  process labels:')

        print('*' * 10 + "process data!" + '*' * 10)
        gtr = np.copy(self.gtr)
        initial = gtr[:, 0:10]
        label = gtr[:, 10:70]
        subOutput = self.subOutput
        print('*' * 10 + "transform MinMax data!" + '*' * 10)
        initialMin, initialMax = self.fieldTransformer(initial, 3)
        labelMin, labelMax = self.fieldTransformer(label, 3)
        print('*' * 10 + "ending transforming MinMax data!" + '*' * 10)
        print('*' * 10 + "complete processing data!" + '*' * 10)

        print('*' * 10 + "saving data!" + '*' * 10)
        self.save(subOutput, name="input", directory=directory, subDirectory="dataset")
        self.save(initialMin, name="initialMin", directory=directory, subDirectory="dataset")
        self.save(initialMax, name="initialMax", directory=directory, subDirectory="dataset")
        self.save(labelMin, name="labelMin", directory=directory, subDirectory="dataset")
        self.save(labelMax, name="labelMax", directory=directory, subDirectory="dataset")
        print('*' * 10 + "end processing data!" + '*' * 10)

    def featureLabel(self, directory='test'):
        # ---------------------- main network ----------------------      
        logging.info('  process labels:')
        trajectory = np.copy(self.gtr)
        # densityLabel_last = (batch, 100, 100) in last 10 timesteps
        contDensityLabel_last = self.trajectoryDivdeMerge(trajectory, 60, 70)
        # densityLabel_last = self.intervalDensity(trajectory, trajectory.shape[1]-10, trajectory.shape[1])
        self.save(contDensityLabel_last, name='label_mainnet_last_cont', directory=directory, subDirectory='fushion')
        # densityLabel_avg = (batch, 100, 100) in last 60 timesteps
        # densityLabel_avg = self.intervalDensity(trajectory, trajectory.shape[1]-60, trajectory.shape[1])
        contDensityLabel_avg = self.trajectoryDivdeMerge(trajectory, 10, 70)
        self.save(contDensityLabel_avg, name='label_mainnet_avg_cont', directory=directory, subDirectory='fushion')

        logging.info('')
        logging.info('  process features:')
        # desntiyFeature = (batch, 100, 100) in first 10 timesteps
        contDesntiyFeature = self.trajectoryDivdeMerge(trajectory, 0, 10)
        self.save(contDesntiyFeature, name='data_init_cont', directory=directory, subDirectory='fushion')
        # tasklist = (batch, 60, 15, 5)
        self.mt = self.taskTimeDivde(self.mt)
        self.save(self.mt, name='data_tasks_cont', directory=directory, subDirectory='fushion')
        # subnet output = (batch, 60, 100, 100)
        self.subOutput = self.taskTimeDivde(self.subOutput)
        self.save(self.subOutput, name='data_subnet_cont', directory=directory, subDirectory='fushion')
        logging.info('')
        # subnet Cube output = (batch, 60, 100, 100)
        # self.save(self.subOutputCube, name='data_subnet_cube', directory=directory, subDirectory='fushion')
        # logging.info('')

        # ---------------------- sub network ----------------------
        logging.info('  process subnet label:')
        self.save(self.sl, name="label_subnet", directory=directory, subDirectory='subnet')
        logging.info('  process subnet tasklist:')
        self.save(self.st, name="data_tasklist", directory=directory, subDirectory='subnet')
        # ---------------------- No Fly Zone ----------------------
        logging.info('  process subnet label:')
        self.save(self.nfz, name="data_nfz", directory=directory, subDirectory='fushion')

        print('finish saving')


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.disabled = False
    logging.basicConfig(filename='log.txt', format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info('Started')

    #s = SimulatorNFZ(batch=1, time=600, mapSize=100, taskNum=15, trajectoryTime=370, taskTime=360)
    s = SimulatorNFZ(batch=3000, time=350, mapSize=100, taskNum=15, trajectoryTime=70, taskTime=60)
    startTimeTotal = time.time()
    s.generate()
    # mainList:(1, 360, 15, 5),label:(1, 370, 100, 100), subOutput:(1, 360, 100, 100),subOutputCube(1, 360, 100, 100)
    # rfeature:(1, 100, 100, 2),noFlyZone(1, 100, 100),subLabel(360, 100, 100),subList(360, 15, 5)
    # print("mainList:{0},label:{1}, subOutput:{2},subOutputCube{3}\nrfeature:{4},noFlyZone{5},subLabel{6},subList{7}"
    #       .format(s.mainTaskList.shape, s.trajectors.shape, s.subOutput.shape, s.subOutputCube.shape,
    #               s.Rfeature.shape, s.NFZ.shape, s.subLabel.shape, s.subTaskList.shape))
    logging.info('Simulater Finished')

    logging.info('  generation costs {0} \n'.format(time.time() - startTimeTotal))
    print('avg flying time: ', s.totalFlyingTime / s.totalUavNum)
    logging.info('avg flying time: {0} \n'.format(s.totalFlyingTime / s.totalUavNum))
    print('total tasks number: ', s.totalUavNum)
    logging.info('total tasks number: {0} \n'.format(s.totalUavNum))

    p = Preprocess(mainList=s.mainTaskList, label=s.trajectors,
                   subOutput=s.subOutput, subOutputCube=s.subOutputCube,
                   rfeature=s.Rfeature, noFlyZone=s.NFZ,
                   subLabel=s.subLabel, subList=s.subTaskList)
    p.featureMinMax(directory="minmax")
    #    p.featureLabel(directory='avg_flow')

    logging.info('Finished dataPreprocess')
    print('Finished dataPreprocess')
