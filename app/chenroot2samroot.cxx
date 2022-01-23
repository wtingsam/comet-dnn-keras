#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TEllipse.h"
#include "TF1.h"
#include "TChain.h"
#include "TH1.h"
#include "TH2D.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"
#include "TRandom.h"
#include "TString.h"
#include "TVector2.h"
#include "TSpectrum2.h"
#include "TEllipse.h"

#include "TMinuit.h"

#include "WireManager.hxx"

// Global defined variables
using namespace std;
int DEBUG = 0;
const int MAX_LAYER = 18;
const int MAX_CELL  = 300;
const int IMAGE_H = 64;
const int IMAGE_W = 96; 

/* A function to get ADC sum from charge */
double GetADCSum(double totalCharge){	
    // Experimental function by Katayama san
    TF1 *fADC2ChargeFunction 
        = new TF1("a2c",
		  "5.98739+2.6652*x+0.000573394*x*x-5.21769e-05*x*x*x+3.05897e-07*x*x*x*x-7.54057e-10*x*x*x*x*x+8.60252e-13*x*x*x*x*x*x-3.68603e-16*x*x*x*x*x*x*x",
		  -10,800);// katayamasan*
    // Linear part of katayama function
    TF1 *fLinearADC2ChargeFunction = new TF1("a2cl","2.32429*x-24.5026",10,1e10);    
    // Must be in fC unit for charge
	double q=fLinearADC2ChargeFunction->GetX(totalCharge,0,1e10);
	if(q>32*700) q = 32*700; // This is the limitation of ADC
	return q;
}

/* Main functions */
int main(int argc, char** argv){
     // Read wire mapping
     WireManager::Get().Initialize();
    
     TString input  = argv[1];
     TString output = argv[2];
    
     if(argc!=3){
		 printf("%s <chen.root><sam.root>\n",argv[0]);
		 return 1;
     }
	 // get input file
	 std::vector<int>		*v_cdc_layerId = 0;
	 std::vector<int>		*v_cdc_cellId  = 0;
	 std::vector<double>	*v_cdc_edep    = 0;
	 std::vector<double>	*v_cdc_DOCA    = 0;
	 std::vector<double>        *v_cdc_mc_x = 0; 
	 std::vector<double>        *v_cdc_mc_y = 0; 
	 std::vector<double>        *v_cdc_mc_z = 0;
	 std::vector<double>        *v_cdc_px = 0; 
	 std::vector<double>        *v_cdc_py = 0; 
	 std::vector<double>        *v_cdc_pz = 0;
	 std::vector<int>        *v_cdc_turnId = 0;
		 
	 double	first_px = 0;
	 double	first_py = 0;
	 double	first_pz = 0;
	 double	first_ix = 0;
	 double	first_iy = 0;
	 double	first_iz = 0;
	 int trig = 0;
	 int nTurns = 0;

	 TChain * iChain = new TChain("mc","mc");
	 iChain->Add(input);
	 iChain->SetBranchAddress("trig", &trig);
	 iChain->SetBranchAddress("nTurns", &nTurns);
	 iChain->SetBranchAddress("cdc_layerId", &v_cdc_layerId);
	 iChain->SetBranchAddress("cdc_cellId", &v_cdc_cellId);
	 iChain->SetBranchAddress("cdc_edep", &v_cdc_edep);
	 iChain->SetBranchAddress("cdc_DOCA", &v_cdc_DOCA);
	 iChain->SetBranchAddress("first_px", &first_px);
	 iChain->SetBranchAddress("first_py", &first_py);
	 iChain->SetBranchAddress("first_pz", &first_pz);
	 iChain->SetBranchAddress("first_x", &first_ix);
	 iChain->SetBranchAddress("first_y", &first_iy);
	 iChain->SetBranchAddress("first_z", &first_iz);
	 iChain->SetBranchAddress("cdc_x", &v_cdc_mc_x);
	 iChain->SetBranchAddress("cdc_y", &v_cdc_mc_y);
	 iChain->SetBranchAddress("cdc_z", &v_cdc_mc_z);
	 iChain->SetBranchAddress("cdc_px", &v_cdc_px);
	 iChain->SetBranchAddress("cdc_py", &v_cdc_py);
	 iChain->SetBranchAddress("cdc_pz", &v_cdc_pz);
	 iChain->SetBranchAddress("cdc_turnId", &v_cdc_turnId);

	 TFile *out = new TFile(output,"recreate");
	 TTree *tree = new TTree("t","Root from chen's NewSDRes");
	 
	 float fFileID_out;
     float fEventNumber_out;
     float fTriggerNumber_out;
	 float fInitialPt_out;
	 float fInitialPz_out;
	 float fInitialX_out;
	 float fInitialY_out;
	 float fInitialZ_out;
	 
	 std::vector<int> fIsSignal_v_out;
	 std::vector<int> fTurnID_v_out;
	 std::vector<int> fWireID_v_out;
	 std::vector<int> fLayerID_v_out;
	 std::vector<int> fCellID_v_out;
	 std::vector<double> fX_v_out;
	 std::vector<double> fY_v_out;
	 std::vector<double> fZ_v_out;
	 std::vector<double> fPx_v_out;
	 std::vector<double> fPy_v_out;
	 std::vector<double> fPz_v_out;
	 std::vector<double> fXe0_v_out;
	 std::vector<double> fYe0_v_out;
	 std::vector<double> fDriftTime_out;
	 std::vector<double> fADC_out;
	 std::vector<int>    fOverLapped_v_out;
     // Define branch
	 tree->Branch("fileID"    ,&fFileID_out);
     tree->Branch("iev"       ,&fEventNumber_out);
     tree->Branch("tn"        ,&fTriggerNumber_out);
	 tree->Branch("isSignal"  ,&fIsSignal_v_out);
	 tree->Branch("turnID"    ,&fTurnID_v_out);
	 tree->Branch("wire"      ,&fWireID_v_out);
	 tree->Branch("layerID"   ,&fLayerID_v_out);
	 tree->Branch("cellID"    ,&fCellID_v_out);
	 tree->Branch("x"	      ,&fX_v_out);
	 tree->Branch("y"	      ,&fY_v_out);
	 tree->Branch("z"	      ,&fZ_v_out);
	 tree->Branch("px"	      ,&fPx_v_out);
	 tree->Branch("py"	      ,&fPy_v_out);
	 tree->Branch("pz"	      ,&fPz_v_out);
	 tree->Branch("xe0"	      ,&fXe0_v_out);
	 tree->Branch("ye0"	      ,&fYe0_v_out);
	 tree->Branch("driftTime" ,&fDriftTime_out);
	 tree->Branch("ADC"	      ,&fADC_out);
	 
	 tree->Branch("olhit"  ,&fOverLapped_v_out);
	 tree->Branch("ipt" ,&fInitialPt_out);
	 tree->Branch("ipz" ,&fInitialPz_out);
	 tree->Branch("ix"  ,&fInitialX_out);
	 tree->Branch("iy"  ,&fInitialY_out);
	 tree->Branch("iz"  ,&fInitialZ_out);
	 // Event loop 
	 int trigger_number = 0;
	 for(int iev=0;iev<iChain->GetEntries();iev++){
		 iChain->GetEntry(iev);
		 //if(!trig) continue;
		 
		 // Fill labels
		 const int nhits = (int)v_cdc_layerId->size();
		 if(nhits<30)continue;
		 
		 fFileID_out = 0;
		 fEventNumber_out=iev;
		 fTriggerNumber_out=trigger_number;

		 int max_index = 0;
		 double max_mom = -1e9;
		 for(int i=0;i<nhits;i++){
			 // Get layer and cells
             int layerID = v_cdc_layerId->at(i);
			 int cellID = v_cdc_cellId->at(i);
			  // Push back data for each hit
		 	 fIsSignal_v_out.push_back(1);
		 	 fTurnID_v_out.push_back(v_cdc_turnId->at(i));
		 	 fWireID_v_out.push_back(-1);
		 	 fLayerID_v_out.push_back(layerID);
		 	 fCellID_v_out.push_back(cellID);
		 	 fX_v_out.push_back(v_cdc_mc_x->at(i));
		 	 fY_v_out.push_back(v_cdc_mc_y->at(i));
		 	 fZ_v_out.push_back(v_cdc_mc_z->at(i));
		 	 //TODOS add these
		 	 fPx_v_out.push_back(v_cdc_px->at(i));
		 	 fPy_v_out.push_back(v_cdc_py->at(i));
		 	 fPz_v_out.push_back(v_cdc_pz->at(i));
			 double x0 = WireManager::Get().GetSenseWireXPosRO(layerID+1,cellID);
			 double y0 = WireManager::Get().GetSenseWireYPosRO(layerID+1,cellID);
		 	 fXe0_v_out.push_back(x0);
		 	 fYe0_v_out.push_back(y0);
		 	 fDriftTime_out.push_back(v_cdc_DOCA->at(i)/0.0023); // To make it nano second
		 	 fADC_out.push_back(GetADCSum(v_cdc_edep->at(i)));
		 }
		 fInitialPt_out = sqrt(first_px*first_px+first_py*first_py);
		 fInitialPz_out = first_pz;
		 fInitialX_out = first_ix;
		 fInitialY_out = first_iy;
		 fInitialZ_out = first_iz;

		 trigger_number++;

		 // //Fill up the tree
		 tree->Fill();
		 // Clear out
		 fIsSignal_v_out.clear();
		 fTurnID_v_out.clear();
		 fWireID_v_out.clear();
		 fLayerID_v_out.clear();
		 fCellID_v_out.clear();
		 fX_v_out.clear();
		 fY_v_out.clear();
		 fZ_v_out.clear();		
		 //TODOS add these
		 fPx_v_out.clear();
		 fPy_v_out.clear();
		 fPz_v_out.clear();
		 fXe0_v_out.clear();
		 fYe0_v_out.clear();
		 fDriftTime_out.clear();
		 fADC_out.clear();
		 fOverLapped_v_out.clear();
				
	 }
	 tree->Write();
}
