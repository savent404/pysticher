from sticher import Sticher
from frameFactory import SimulateFrameFactory

if __name__ == '__main__':
    ff = SimulateFrameFactory()

    sticher = Sticher([ff])
    sticher.run()
    sticher.download_map('./result.jpg')
