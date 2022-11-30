import os
import sys
import pkg_resources
import time
here = os.path.dirname(os.path.abspath(__file__))

# if vulkanese isn't installed, check for a development version parallel to Loiacono repo ;)
if "vulkanese" not in [pkg.key for pkg in pkg_resources.working_set]:
    sys.path = [os.path.join(here, "..", "vulkanese", "vulkanese")] + sys.path

from vulkanese import *
from loiacono import *

class Loiacono_GPU(ComputeShader, Loiacono):
    def __init__(
        self,
        device,
        signalLength,
        fprime,
        multiple, 
        constantsDict = {},
        DEBUG=False,
        buffType="float",
        memProperties=0
        | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    ):

        self.midiRange = midiend-midistart
        constantsDict["multiple"] = multiple
        constantsDict["SIGNAL_LENGTH"] = signalLength
        constantsDict["PROCTYPE"] = buffType
        constantsDict["TOTAL_THREAD_COUNT"] = signalLength * len(fprime)
        constantsDict["LG_WG_SIZE"] = 7
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]
        self.dim2index = {}
        

        # device selection and instantiation
        self.instance = device.instance
        self.device = device
        self.constantsDict = constantsDict
        self.numSubgroups = signalLength * len(fprime) / self.device.subgroupSize
        self.numSubgroupsPerFprime = int(self.numSubgroups / len(fprime))

        # declare buffers. they will be in GPU memory, but visible from the host (!)
        buffers = [
            StorageBuffer(
                device=self.device,
                name="x",
                memtype=buffType,
                qualifier="readonly",
                dimensionVals=[2**15], # always 32**3
                memProperties=memProperties,
            ),
            StorageBuffer(
                device=self.device,
                name="Li1",
                memtype=buffType,
                dimensionVals=[len(fprime), self.device.subgroupSize**2],
            ),
            StorageBuffer(
                device=self.device,
                name="Lr1",
                memtype=buffType,
                dimensionVals=[len(fprime), self.device.subgroupSize**2],
            ),
            StorageBuffer(
                device=self.device,
                name="Li0",
                memtype=buffType,
                dimensionVals=[len(fprime), self.device.subgroupSize],
            ),
            StorageBuffer(
                device=self.device,
                name="Lr0",
                memtype=buffType,
                dimensionVals=[len(fprime), self.device.subgroupSize],
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
            #DebugBuffer(
            #    device=self.device,
            #    name="allShaders",
            #    memtype=buffType,
            #    dimensionVals=[constantsDict["TOTAL_THREAD_COUNT"]],
            #),
        ]
        
        # Create a compute shader
        # Compute Stage: the only stage
        shader_basename="shaders/loiacono",
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
        self.getHarmonicPattern()

    def debugRun(self):
        vstart = time.time()
        self.run()
        vlen = time.time() - vstart
        self.absresult = self.L
        print("vlen " + str(vlen))
        # return self.sumOut.getAsNumpyArray()


if __name__ == "__main__":

    infile = sys.argv[1]
    multiple = 10
    # load the wav file
    y, sr = librosa.load(infile, sr=None)
    
    midiIndices, fprime = getMidiFprime(
        subdivisionOfSemitone=1.0,
        midistart=30,
        midiend=110,
        sr=sr,
    )
    
    # generate a Loiacono based on this SR
    linst = Loiacono(
        fprime = fprime,
        multiple=multiple
    )
    
    # get a section in the middle of sample for processing
    z = y[int(len(y) / 2) : int(len(y) / 2 + linst.DTFTLEN)]
    linst.debugRun(z)
    print(linst.selectedNote)
    
    # begin GPU test
    instance = Instance(verbose=False)
    devnum = 0
    device = instance.getDevice(devnum)
    
    z = y[int(len(y) / 2) : int(len(y) / 2 + 2**15)]
    
    # coarse detection
    linst_gpu = Loiacono_GPU(
        device = device,
        signalLength = 2**15,
        fprime = fprime,
        multiple = linst.multiple,
        constantsDict = {},
    )
    
    linst_gpu.x.setBuffer(z)
    for i in range(10):
        linst_gpu.debugRun()
    #linst_gpu.dumpMemory()
    readstart = time.time()
    linst_gpu.absresult = linst_gpu.L.getAsNumpyArray()
    print("Readtime " + str(time.time()- readstart))
    v2time = time.time()
    linst_gpu.findNote()
    linst.findNote()
    print("v2rt " + str(time.time() - v2time))
    print(linst_gpu.selectedNote)
    print(linst.selectedNote)
    
    precision = 10
    midiIndicesToCheck = np.arange(linst_gpu.selectedNote-0.5, linst_gpu.selectedNote+0.5, step = 1.0/precison)
    print(midiIndicesToCheck)
    die
    # fine detection
    linst_gpu_fine = Loiacono_GPU(
        device = device,
        signalLength = 2**15,
        fprime = fprime_fine,
        multiple = 30,
        constantsDict = {},
    )
    
    
    print(len(linst_gpu.absresult))
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.plot(linst.midiIndices, linst_gpu.notesPadded)
    ax2.plot(linst.midiIndices, linst.notesPadded)
    
    plt.show()
    
    #print(json.dumps(list(ar), indent=2))
