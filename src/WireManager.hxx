#ifndef _WIREMANAGER_HXX_
#define _WIREMANAGER_HXX_

#include "WireConfig.hxx"

class WireManager : public WireConfig {
    public:
        ~WireManager(){}

        static WireManager &Get();
	// The output layer is without guard layer
	// i.e. [0,17]
	void GetWireID2LayerCell(int wireId, int& layer, int& cell){
	    std::pair<int,int> laycell = fCh2LayCellMap[wireId];
	    layer = laycell.first;
	    cell  = laycell.second;
	}
	// The input layer should be considering without guard layer
	int GetChannelId(int layer, int cell){
	    std::pair <int,int> layCell = std::make_pair(layer,cell);
	    return fLayCell2ChMap[layCell];
	}

	/// Get information related to the boards
	std::vector<int> GetChListOnASD(int ASD){
	    return fASD2WireIdList[ASD];
	}
	int GetBoardID(int layer, int cell){   return fBoardID[layer][cell];  }
	int GetASDID(int layer, int cell){   return fASD[layer][cell];   }
	int GetASDID(int wireId){
	    int layer,cell;
	    GetWireID2LayerCell(wireId,layer,cell);
	    return GetASDID(layer+1,cell);
	}

        /// Get Wire position at Endplates	
        double GetSenseWireXPosHV(int layer, int wire){if (CheckLayer(layer)||CheckWire(wire)) return -1; else return fXhv[layer][wire]; }
        double GetSenseWireYPosHV(int layer, int wire){if (CheckLayer(layer)||CheckWire(wire)) return -1; else return fYhv[layer][wire]; }
        double GetSenseWireZPosHV(int layer){ if (CheckLayer(layer)) return -1; else return -fCDClength[layer]/2; }
        double GetSenseWireXPosCen(int layer, int wire){if (CheckLayer(layer)||CheckWire(wire)) return -1; else return fXc[layer][wire]; }
        double GetSenseWireYPosCen(int layer, int wire){if (CheckLayer(layer)||CheckWire(wire)) return -1; else return fYc[layer][wire]; }
        double GetSenseWireXPosRO(int layer, int wire){if (CheckLayer(layer)||CheckWire(wire)) return -1; else return fXro[layer][wire]; }
        double GetSenseWireYPosRO(int layer, int wire){if (CheckLayer(layer)||CheckWire(wire)) return -1; else return fYro[layer][wire]; }
        double GetSenseWireZPosRO(int layer){ if (CheckLayer(layer)) return -1; else return  fCDClength[layer]/2; }
	
        /// Geometry
        int GetNumSenseWires(void){ return fNumSenseWire; }
        int GetNumSenseWires(int layer){ return fMaxSenseWirePerLayer[layer]; }
        int GetNumFieldWires(void){ return fNumFieldWire; }

        bool IsWire(int layer, int wire);


        int CheckLayer(int ly){
            if (ly>=MAX_SENSE_LAYER){
                if(fDebug)fprintf(stderr,"layerID %d is larger than MAX_SENSE_LAYER = %d\n",ly,MAX_SENSE_LAYER);
                return 1;
            }
            else if (ly<0){
                if(fDebug)fprintf(stderr,"Invalid layerID %d < 0\n", ly);
                return -1;
            }
            else{
                return 0;
            }
        }

        int CheckWire(int wi){
            if (wi>=MAX_WIREpL){
                if(fDebug)fprintf(stderr,"wireID %d is larger than MAX_WIREpL = %d\n",wi,MAX_WIREpL);
                return 1;
            }
            else if (wi<0){
                if(fDebug)fprintf(stderr,"Invalid wireID %d < 0\n", wi);
                return -1;
            }
            else{
                return 0;
            }
        }

    private:
        static WireManager *fWireManager;
        WireManager(){}; //Don't Implenment
        WireManager(WireManager const&); //Don't Implenment
        void operator=(WireManager const&); //Don't Implenment


};




#endif
