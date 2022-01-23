#include "WireManager.hxx"

WireManager *WireManager::fWireManager = NULL;

WireManager &WireManager::Get()
{
    if ( !fWireManager ){//make sure only being created one time
        fWireManager = new WireManager();
    }
    return *fWireManager;
}


bool WireManager::IsWire(int layer, int wire)
{
    //later
}

