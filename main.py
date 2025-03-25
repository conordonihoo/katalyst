from typing import List

from utils.consts import Constants
from utils.ecef import *
from utils.eci import *

def parseCsv(csv_path: str) -> List[EciState]:
    return []

def main() -> None:
    # get ground station locations in ECEF
    r_ecef_gs1 = geodeticToEcef(*Constants.LLA_GS1)
    r_ecef_gs2 = geodeticToEcef(*Constants.LLA_GS2)
    return

if __name__ == '__main__':
    main()
