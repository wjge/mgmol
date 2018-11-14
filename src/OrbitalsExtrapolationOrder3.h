// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. 
// Written by J.-L. Fattebert, D. Osei-Kuffuor and I.S. Dunn.
// LLNL-CODE-743438
// All rights reserved. 
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_ORBITALSEXTRAPOLATIONORDER3_H
#define MGMOL_ORBITALSEXTRAPOLATIONORDER3_H

#include "OrbitalsExtrapolation.h"
#include "LocGridOrbitals.h"

class OrbitalsExtrapolationOrder3 : public OrbitalsExtrapolation
{
private:
    LocGridOrbitals* initial_orbitals_minus2_;
    LocGridOrbitals* orbitals_minus2_;
public:
    
    OrbitalsExtrapolationOrder3()
    {
        initial_orbitals_minus2_=0;
        orbitals_minus2_=0;
    };
    
    ~OrbitalsExtrapolationOrder3()
    {
        if( orbitals_minus2_!=0 )
        {
            delete orbitals_minus2_;
            orbitals_minus2_=0;
        }
        if( initial_orbitals_minus2_!=0 )
        {
            delete initial_orbitals_minus2_;
            initial_orbitals_minus2_=0;
        }
    }

    void extrapolate_orbitals(LocGridOrbitals** orbitals, 
                              LocGridOrbitals* new_orbitals);


    void clearOldOrbitals()
    {
        OrbitalsExtrapolation::clearOldOrbitals();
    
        if( orbitals_minus2_!=0 )
        {
            delete orbitals_minus2_;
            orbitals_minus2_=0;
        }
    }
};

#endif
