#ifndef _WIRECONFIG_HXX_
#define _WIRECONFIG_HXX_

#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include <map>
class WireConfig{

    public:
        WireConfig();
        ~WireConfig();

	void Initialize(TString path_textfile="info/chanmap_20160814.txt",double rotate=0, int debug=0);
        /// Path for info
        virtual void CheckPath(void);

    private:
        virtual void SetPath(TString path){ fWireMapPath=path; }
        /// The path for obtaining the wire map root file
        TString  fWireMapPath;
        TFile *fRootFile;
        TTree *fTree;
	Int_t fAngle;
	
	/// Read a mapping at info/
        void ReadWireMap(void);//modification

	/// Rotate the setup by an angle (Easy for linear fitting)
	void Rotation(double x1, double y1, double& x2, double& y2, double ang){
	    const double pi=3.141592;
	    double A = pi*ang/180;
	    x2 =  x1*cos(A)+y1*sin(A);
	    y2 = -x1*sin(A)+y1*cos(A);
	}
	
    protected:

	// channel id to layer cell id
	std::map < int, std::pair < int, int > > fCh2LayCellMap;
	std::map < std::pair < int, int >, int > fLayCell2ChMap;

	// asd to layer wire
	std::map < int, std::vector<std::pair<int, int> > > fASD2LayCellListMap;
	std::map < int, std::vector<int> > fASD2WireIdList;
	
	// Configuration variable
        int MAX_WIREpL;
        int MAX_SENSE_LAYER;

        int MAX_LAYER;
	int MAX_WIREID;

        //Geometry
        double *fCDClength;
        double **fXhv;
        double **fYhv;
        double **fXro;
        double **fYro;
        double **fXc;
        double **fYc;
	int **fASD;
	int **fBoardID;

	int *fMaxWirePerLayer;
	int *fMaxSenseWirePerLayer;
	
        double **fXhv_all;
        double **fYhv_all;
        double **fXro_all;
        double **fYro_all;

        int fNumSenseWire_noguard;
        int fNumSenseWire;
        int fNumFieldWire;

	// debug
	int fDebug;
};




#endif
