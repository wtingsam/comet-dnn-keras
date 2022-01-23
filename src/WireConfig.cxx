#include "TFile.h"
#include <sstream>
#include <fstream>
#include "WireConfig.hxx"
#include <stdlib.h>

WireConfig::WireConfig(){}
WireConfig::~WireConfig(){}

void WireConfig::Initialize(TString path_textfile, double rotate, int debug)
{
    fDebug = debug;
    fAngle = rotate;
	
    MAX_WIREpL	    = 306;
    MAX_SENSE_LAYER = 20;
    MAX_LAYER	    = 39;
    MAX_WIREID	    = 612;

    //1D array
    fCDClength = (double*)malloc(sizeof(double)*MAX_SENSE_LAYER);
    fMaxSenseWirePerLayer = (int*)malloc(sizeof(double)*MAX_SENSE_LAYER);

    fASD = (int**)malloc(sizeof(double*)*MAX_SENSE_LAYER);
    fBoardID = (int**)malloc(sizeof(double*)*MAX_SENSE_LAYER);
    fXhv = (double**)malloc(sizeof(double*)*MAX_SENSE_LAYER);
    fYhv = (double**)malloc(sizeof(double*)*MAX_SENSE_LAYER);
    fXro = (double**)malloc(sizeof(double*)*MAX_SENSE_LAYER);
    fYro = (double**)malloc(sizeof(double*)*MAX_SENSE_LAYER);
    fXc  = (double**)malloc(sizeof(double*)*MAX_SENSE_LAYER);
    fYc  = (double**)malloc(sizeof(double*)*MAX_SENSE_LAYER);

    for(int i=0;i<MAX_SENSE_LAYER;i++){
	fASD[i] = (int*)malloc(sizeof(int)*MAX_WIREpL);
	fBoardID[i] = (int*)malloc(sizeof(int)*MAX_WIREpL);
        fXhv[i]	= (double*)malloc(sizeof(double)*MAX_WIREpL);
        fYhv[i]	= (double*)malloc(sizeof(double)*MAX_WIREpL);
        fXro[i]	= (double*)malloc(sizeof(double)*MAX_WIREpL);
        fYro[i]	= (double*)malloc(sizeof(double)*MAX_WIREpL);
        fXc[i]	= (double*)malloc(sizeof(double)*MAX_WIREpL);
        fYc[i]	= (double*)malloc(sizeof(double)*MAX_WIREpL);
    }

    fXhv_all = (double**)malloc(sizeof(double*)*MAX_LAYER);
    fYhv_all = (double**)malloc(sizeof(double*)*MAX_LAYER);
    fXro_all = (double**)malloc(sizeof(double*)*MAX_LAYER);
    fYro_all = (double**)malloc(sizeof(double*)*MAX_LAYER);
    
    for(int i=0;i<MAX_LAYER;i++){
        fXhv_all[i] = (double*)malloc(sizeof(double)*MAX_WIREID);
        fYhv_all[i] = (double*)malloc(sizeof(double)*MAX_WIREID);
        fXro_all[i] = (double*)malloc(sizeof(double)*MAX_WIREID);
        fYro_all[i] = (double*)malloc(sizeof(double)*MAX_WIREID);
    }

    fWireMapPath = path_textfile;
    fNumSenseWire        = 0;
    fNumFieldWire        = 0;
    ReadWireMap();
}

void WireConfig::CheckPath(void){

    if(fWireMapPath!=""){
        std::cout << "## "<< fWireMapPath << std::endl;
    }else{
        std::cerr << "Did not set Path" << std::endl;
    }
}  //for debug


void WireConfig::ReadWireMap(void)
{
    ifstream wireMapFile(fWireMapPath);
    if(!wireMapFile.is_open()){
      std::cerr << "## ERROR cannot open WireMapPath file: \"" << fWireMapPath <<"\""<< std::endl;;
        return ;
    }
    
    Double_t LayerLength,xhv,yhv,x0,y0,xro,yro,layer,wire,LayerID,isSenseWire,CellID,BoardID,BrdLayID,BrdLocID,ChanID;
    Double_t buf[20];
    std::string line;

    while(getline(wireMapFile,line)){
	std::istringstream iss(line);

	iss>>LayerLength>>layer>>wire>>xhv>>yhv>>x0>>y0>>xro>>yro>>LayerID>>isSenseWire>>CellID>>BoardID>>BrdLayID>>BrdLocID>>ChanID;

        if(isSenseWire){
	    std::pair <int,int> layCell;
            int l_tmp = (int)LayerID;
            int c_tmp = (int)CellID;
	    if(l_tmp>0 && l_tmp<MAX_SENSE_LAYER-1){
	    	layCell = std::make_pair(l_tmp-1,c_tmp);
	    	fCh2LayCellMap[fNumSenseWire_noguard] = layCell;
		fLayCell2ChMap[layCell] = fNumSenseWire_noguard;
		fASD[l_tmp][c_tmp]=(int)ChanID%8 + (int)BoardID*6;
		fASD2WireIdList[fASD[l_tmp][c_tmp]].push_back(fNumSenseWire_noguard);

	    	fNumSenseWire_noguard++;
	    }
	    fBoardID[l_tmp][c_tmp] = (int)BoardID;
	    
	    //    fASD2LayCellListMap[f

            //Length of the CDC
	    fMaxSenseWirePerLayer[l_tmp]++;
	    fCDClength[l_tmp]=LayerLength;
            fNumSenseWire++;	    
	    //printf("wireId %d layer %4d cell %4d boardid %d channelID %d asdid %4d \n", fNumSenseWire,l_tmp,c_tmp,(int)BoardID,(int)ChanID,fASD[l_tmp][c_tmp]);
            fXhv[l_tmp][c_tmp]=xhv/10;//[cm]
            fYhv[l_tmp][c_tmp]=yhv/10;//[cm]
            fXro[l_tmp][c_tmp]=xro/10;//[cm]
            fYro[l_tmp][c_tmp]=yro/10;//[cm]
            fXc[l_tmp][c_tmp]=x0/10;//[cm]
            fYc[l_tmp][c_tmp]=y0/10;//[cm]

        }else{
            fNumFieldWire++;
        }
    }
}
