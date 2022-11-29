import os
import sys
import pkg_resources
import time

here = os.path.dirname(os.path.abspath(__file__))
if "vulkanese" not in [pkg.key for pkg in pkg_resources.working_set]:
    sys.path = [os.path.join(here, "..", "vulkanese", "vulkanese")] + sys.path

from vulkanese import *
import numpy as np
                       
here = os.path.dirname(os.path.abspath(__file__))
from loiacono import *

class Loiacono_GPU(ComputeShader):
    def __init__(
        self,
        instance,
        device,
        signalLength,
        fprime,
        m, 
        constantsDict = {},
        DEBUG=False,
        buffType="float",
        shader_basename="shaders/loiacono",
        memProperties=0
        | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    ):

        constantsDict["m"] = m
        constantsDict["SIGNAL_LENGTH"] = signalLength
        constantsDict["PROCTYPE"] = buffType
        constantsDict["LG_WG_SIZE"] = 7
        constantsDict["TOTAL_THREAD_COUNT"] = signalLength * len(fprime)
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]
        self.dim2index = {}

        # device selection and instantiation
        self.instance = instance
        self.device = device
        self.constantsDict = constantsDict
        self.subgroupSize = 32
        self.numSubgroups = signalLength * len(fprime) / self.subgroupSize
        self.numSubgroupsPerFprime = int(self.numSubgroups / len(fprime))


        buffers = [
            StorageBuffer(
                device=self.device,
                name="x",
                memtype=buffType,
                qualifier="readonly",
                dimensionVals=signalLength,
                memProperties=memProperties,
            ),
            StorageBuffer(
                device=self.device,
                name="Li1",
                memtype=buffType,
                dimensionVals=[len(fprime), self.subgroupSize**2],
            ),
            StorageBuffer(
                device=self.device,
                name="Lr1",
                memtype=buffType,
                dimensionVals=[len(fprime), self.subgroupSize**2],
            ),
            DebugBuffer(
                device=self.device,
                name="Li0",
                memtype=buffType,
                dimensionVals=[len(fprime), self.subgroupSize],
            ),
            DebugBuffer(
                device=self.device,
                name="Lr0",
                memtype=buffType,
                dimensionVals=[len(fprime), self.subgroupSize],
            ),
            StorageBuffer(
                device=self.device,
                name="L",
                memtype=buffType,
                qualifier="writeonly",
                dimensionVals=[len(fprime)],
                memProperties=memProperties,
            ),
            StorageBuffer(
                device=self.device,
                name="f",
                memtype=buffType,
                qualifier="readonly",
                dimensionVals=[len(fprime)],
                memProperties=memProperties,
            ),
            UniformBuffer(
                device=self.device,
                name="offset",
                memtype=buffType,
                qualifier="readonly",
                dimensionVals=[1],
                memProperties=memProperties,
            ),
            DebugBuffer(
                device=self.device,
                name="allShaders",
                memtype=buffType,
                dimensionVals=[constantsDict["TOTAL_THREAD_COUNT"]],
            ),
        ]
        
        # Compute Stage: the only stage
        ComputeShader.__init__(
            self,
            sourceFilename=os.path.join(
                here, shader_basename + ".c"
            ),  # can be GLSL or SPIRV
            parent=self.instance,
            constantsDict=self.constantsDict,
            device=self.device,
            name=shader_basename,
            stage=VK_SHADER_STAGE_COMPUTE_BIT,
            buffers=buffers,
            DEBUG=DEBUG,
            dim2index=self.dim2index,
            memProperties=memProperties,
            workgroupCount=[
                int(
                    signalLength * len(fprime)
                    / (
                        constantsDict["THREADS_PER_WORKGROUP"]
                    )
                ),
                1,
                1,
            ],
            compressBuffers=True,
        )
                
        self.f.setBuffer(fprime)
        self.offset.setBuffer(np.zeros((1)))

    def debugRun(self):
        vstart = time.time()
        self.run()
        vlen = time.time() - vstart
        self.absresult = self.L
        print("vlen " + str(vlen))
        # return self.sumOut.getAsNumpyArray()


if __name__ == "__main__":

    infile = sys.argv[1]
    m = 20
    # load the wav file
    y, sr = librosa.load(infile, sr=None)
    
    # generate a Loiacono based on this SR
    linst = Loiacono(
        subdivisionOfSemitone=4.0,
        midistart=30,
        midiend=112,
        sr=sr,
        multiple=m)
    
    # get a section in the middle of sample for processing
    y = y[int(len(y) / 2) : int(len(y) / 2 + linst.DTFTLEN)]
    linst.debugRun(y)
    print(linst.selectedNote)
    #linst.plot()
    
    if(linst.DTFTLEN != 2**15):
        raise Exception("For GPU, DTFTLEN MUST BE 2**15 = 32768 = 32**3. Instead it is " + str(linst.DTFTLEN))
    emptySignal = np.zeros((linst.DTFTLEN))
    
    # begin GPU test
    instance = Instance(verbose=False)
    devnum = 0
    device = instance.getDevice(devnum)
    
    linst_gpu = Loiacono_GPU(
        instance = instance,
        device = device,
        signalLength = linst.DTFTLEN,
        fprime = linst.fprime,
        m = linst.m,
        constantsDict = {},
    )
    
    linst_gpu.x.setBuffer(y)
    for i in range(10):
        linst_gpu.debugRun()
    linst_gpu.dumpMemory()
    
    v2time = time.time()
    ar = linst_gpu.L.getAsNumpyArray()
    print("v2rt " + str(time.time() - v2time))
    
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.plot(linst.midiIndices, ar)
    #ax1.plot(linst_gpu.allShaders.getAsNumpyArray())
    ax2.plot(linst.midiIndices, linst.absresult)
    plt.show()
    
    print(json.dumps(list(ar), indent=2))
