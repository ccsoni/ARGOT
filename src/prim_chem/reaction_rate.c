//=======================================================================
//
//   reactions for the chemical network
//   k01    : HI    + e   -> HII   + 2e
//   k02    : HII   + e   -> HI    + photon
//   k03    : HeI   + e   -> HeII  + 2e
//   k04    : HeII  + e   -> HeI   + photon
//   k05    : HeII  + e   -> HeIII + 2e
//   k06    : HeIII + e   -> HeII  + photon
//   k07    : HI    + e   -> HM    + photon
//   k08    : HM    + HI  -> H2I*  + e
//   k09    : HI    + HII -> H2II  + photon
//   k10    : H2II  + HI  -> H2I*  + HII
//   k11    : H2I   + HI  -> 3HI
//   k12    : H2I   + HII -> H2II  + HI
//   k13    : H2I   + e   -> HI    + HM
//   k14    : H2I   + e   -> 2HI   + e
//   k15    : H2I   + H2I -> H2I   + 2HI
//   k16    : HM    + e   -> HI    + 2e
//   k17    : HM    + HI  -> 2H    + e
//   k18    : HM    + HII -> 2HI
//   k19    : HM    + HII -> H2II  + e
//   k20    : H2II  + e   -> 2HI
//   k21    : H2II  + HM  -> HI    + H2I
//
//=======================================================================

#include <stdio.h>
#include <math.h>

#include "chemistry.h"

double k01(double t){
  
  double tev;
  double logtev, logtev2, logtev3, logtev4, logtev5, logtev6, logtev7, logtev8;

  tev = t*K_to_eV;
  logtev  = log(tev);
  logtev2 = logtev*logtev;
  logtev3 = logtev2*logtev;
  logtev4 = logtev3*logtev;
  logtev5 = logtev4*logtev;
  logtev6 = logtev5*logtev;
  logtev7 = logtev6*logtev;
  logtev8 = logtev7*logtev;

  //  if(tev > 0.8) {
   return (exp(-32.71396786375 
	       + 13.53655609057*logtev 
	       - 5.739328757388*logtev2
	       + 1.563154982022*logtev3
	       - 0.2877056004391*logtev4
	       + 0.03482559773736999*logtev5
	       - 0.00263197617559*logtev6
	       + 0.0001119543953861*logtev7
	       - 2.039149852002e-6*logtev8));
   //  }else{
   //    return 0.e0;
   //  }

}


double k02(double t)
{
#ifndef __CASE_B__
  double tev;
  double logtev,logtev2,logtev3,logtev4,logtev5,logtev6,logtev7,logtev8,logtev9;

  tev = t*K_to_eV;
  logtev  = log(tev);
  logtev2 = logtev*logtev;
  logtev3 = logtev2*logtev;
  logtev4 = logtev3*logtev;
  logtev5 = logtev4*logtev;
  logtev6 = logtev5*logtev;
  logtev7 = logtev6*logtev;
  logtev8 = logtev7*logtev;
  logtev9 = logtev8*logtev;

//  if (t > 5500.0){
    return (exp(-28.61303380689232 
		- 0.7241125657826851*logtev 
		- 0.02026044731984691*logtev2 
		- 0.002380861877349834*logtev3 
		- 0.0003212605213188796*logtev4 
		- 0.00001421502914054107*logtev5 
		+ 4.989108920299513e-6*logtev6 
		+ 5.755614137575758e-7*logtev7 
		- 1.856767039775261e-8*logtev8 
		- 3.071135243196595e-9*logtev9));
//  }else{
//    return (k04(t));
//  }
#else /* __CASE_B__ */

    /* Hui + Gnedin 1997*/ 

    double lambda = 2.0*157807.0/t;
    return (2.753e-14*pow(lambda,1.5)/pow(1.0+pow(lambda/2.740, 0.407),2.242));

#endif 
}



double k03(double t)
{
  double tev;
  double logtev, logtev2, logtev3, logtev4, logtev5, logtev6, logtev7, logtev8;

  tev = t*K_to_eV;
  logtev  = log(tev);
  logtev2 = logtev*logtev;
  logtev3 = logtev2*logtev;
  logtev4 = logtev3*logtev;
  logtev5 = logtev4*logtev;
  logtev6 = logtev5*logtev;
  logtev7 = logtev6*logtev;
  logtev8 = logtev7*logtev;

  if (tev > 0.8) {
    return (exp(-44.09864886561001 
		+ 23.91596563469*logtev 
		- 10.75323019821*logtev2 
		+ 3.058038757198*logtev3 
		- 0.5685118909884001*logtev4 
		+ 0.06795391233790001*logtev5 
		- 0.005009056101857001*logtev6 
		+ 0.0002067236157507*logtev7 
		- 3.649161410833e-6*logtev8));
  }else{
    return (0.0);
  }

}

double k04(double t) 
{
#ifndef __CASE_B__
  double tev;
  
  tev = t*K_to_eV;

  if (tev > 0.8) {
    return (1.54e-9*(1.+0.3/exp(8.099328789667/tev)) 
	    / (exp(40.49664394833662/tev)*pow(tev,1.5)) 
	    + 3.92e-13/pow(tev,0.6353));
  }else{
    return 3.92e-13/pow(tev,0.6353);
  }
#else /* __CASE_B__ */
  if(t>500.3){ // if log10(T) > 2.69923
    /* Hui + Gnedin */
    double lambda;
    lambda = 2.0*285335.0/t;
    return (1.26e-14*pow(lambda,0.75));
  }else{
    /* Fitting by K. Yoshikawa*/
    double logt;
    
    logt = log10(t);
    return (1.12e-10 - 2.1e-11*logt)/sqrt(t);
  }
#endif
}

double k05(double t)
{
  double tev;
  double logtev, logtev2, logtev3, logtev4, logtev5, logtev6, logtev7, logtev8;

  tev = t*K_to_eV;
  logtev  = log(tev);
  logtev2 = logtev*logtev;
  logtev3 = logtev2*logtev;
  logtev4 = logtev3*logtev;
  logtev5 = logtev4*logtev;
  logtev6 = logtev5*logtev;
  logtev7 = logtev6*logtev;
  logtev8 = logtev7*logtev;

  if (tev > 0.8) {
    return (exp(-68.71040990212001 
		+ 43.93347632635*logtev 
		- 18.48066993568*logtev2 
		+ 4.701626486759002*logtev3 
		- 0.7692466334492*logtev4 
		+ 0.08113042097303*logtev5 
		- 0.005324020628287001*logtev6 
		+ 0.0001975705312221*logtev7 
		- 3.165581065665e-6*logtev8));
  }else{
    return (0.0);
  }
}

double k06(double t)
{
#ifndef __CASE_B__
  return (3.36e-10/sqrt(t)/pow((t/1.e3),0.2)/(1.0+pow((t/1.e6),0.7)));
#else /* __CASE_B__ */
  double lambda = 2.0*631515.0/t;
  return (1.3765e-14*pow(lambda,1.5)/pow(1.0+pow(lambda/2.740, 0.407),2.242));
#endif
}

double k07(double t)
{
  double tev;

  tev = t*K_to_eV;

  return (6.77e-15*pow(tev,0.8779));

  // Galli & Palla (1998)
  // return (1.4e-18*pow(t,0.928)*exp(-t/1.62e4));
}

double k08(double t) 
{
  double tev;
  double logtev, logtev2, logtev3, logtev4, logtev5, logtev6, logtev7;

  tev = t*K_to_eV;
  logtev  = log(tev);
  logtev2 = logtev*logtev;
  logtev3 = logtev2*logtev;
  logtev4 = logtev3*logtev;
  logtev5 = logtev4*logtev;
  logtev6 = logtev5*logtev;
  logtev7 = logtev6*logtev;

  if(tev > 0.1) {
    return (exp(-20.06913897587003 
		+ 0.2289800603272916*logtev 
		+ 0.03599837721023835*logtev2 
		- 0.004555120027032095*logtev3 
		- 0.0003105115447124016*logtev4 
		+ 0.0001073294010367247*logtev5 
		- 8.36671960467864e-6*logtev6 
		+ 2.238306228891639e-7*logtev7));
  }else{
    return (1.43e-9);
  }

  // Galli & Palla (1998)
  // if (t>3.e2) {
  //    return (4.0e-9*pow(t,-0.17));
  // }else{
  //    return (1.5e-9);
  // }
}

double k09(double t)
{
  if(t > 6.7e+3) {
    return (5.81e-16*pow((t/56200.0),(-0.6657*log10(t/56200.0))));
  }else{
    return (1.85e-23*pow(t,1.8));
  }
}

double k10(double t)
{
  return (6.0e-10);
}

double k11(double t)
{
  double tev;

  tev = t*K_to_eV;
  if(tev > 0.3) {
    return (1.0670825e-10*pow(tev,2.012)
	    /(exp(4.463/tev)*pow((1.0+0.2472*tev),3.512)));
  }else{
    return (0.0);
  }
  
}

double k12(double t)
{
  double tev;
  double logtev, logtev2, logtev3, logtev4, logtev5, logtev6, logtev7, logtev8;

  tev = t*K_to_eV;
  logtev  = log(tev);
  logtev2 = logtev*logtev;
  logtev3 = logtev2*logtev;
  logtev4 = logtev3*logtev;
  logtev5 = logtev4*logtev;
  logtev6 = logtev5*logtev;
  logtev7 = logtev6*logtev;
  logtev8 = logtev7*logtev;

  if(tev>0.3) {
    return (exp(-24.24914687731536  
		+ 3.400824447095291*logtev 
		- 3.898003964650152*logtev2 
		+ 2.045587822403071*logtev3 
		- 0.5416182856220388*logtev4 
		+ 0.0841077503763412*logtev5 
		- 0.007879026154483455*logtev6 
		+ 0.0004138398421504563*logtev7 
		- 9.36345888928611e-6*logtev8));
  }else{
    return (0.0);
  }

  // Galli & Palla (1998)
  // if(t>1.0e4) {
  //   return (1.5e-10*exp(-1.4e+4/t));
  // }else{
  //   return (3.0e-10*exp(-2.105e+4/t));
  // }

}

double k13(double t)
{
  return (0.0);

  // Galli & Palla (1998)
  // if(t>2.0e3) {
  //   return (2.7e-8*pow(t,-1.27)*exp(-4.3e+4/t));
  // }else{
  //   return (0.0);
  // }
}

double k14(double t)
{
  double tev;

  tev = t*K_to_eV;

  if(tev > 0.3){
    return (4.38e-10*exp(-102000.0/t)*pow(t,0.35));
  }else{
    return (0.0);
  }
}

double k15(double t)
{
  return (0.0);
}

double k16(double t)
{
  double tev;
  double logtev, logtev2, logtev3, logtev4, logtev5, logtev6, logtev7, logtev8;

  tev = t*K_to_eV;
  logtev  = log(tev);
  logtev2 = logtev*logtev;
  logtev3 = logtev2*logtev;
  logtev4 = logtev3*logtev;
  logtev5 = logtev4*logtev;
  logtev6 = logtev5*logtev;
  logtev7 = logtev6*logtev;
  logtev8 = logtev7*logtev;

  if(tev > 0.04) {
    return (exp(-18.01849334273 
		+ 2.360852208681*logtev 
		- 0.2827443061704*logtev2 
		+ 0.01623316639567*logtev3 
		- 0.03365012031362999*logtev4 
		+ 0.01178329782711*logtev5 
		- 0.001656194699504*logtev6 
		+ 0.0001068275202678*logtev7 
		- 2.631285809207e-6*logtev8));
  }else{
    return (0.0);
  }
}

double k17(double t)
{
  double tev;
  double logtev,logtev2,logtev3,logtev4,logtev5,logtev6,logtev7,logtev8,logtev9;

  tev = t*K_to_eV;
  logtev  = log(tev);
  logtev2 = logtev*logtev;
  logtev3 = logtev2*logtev;
  logtev4 = logtev3*logtev;
  logtev5 = logtev4*logtev;
  logtev6 = logtev5*logtev;
  logtev7 = logtev6*logtev;
  logtev8 = logtev7*logtev;
  logtev9 = logtev8*logtev;

  if(tev > 0.1) {
    return (exp(-20.37260896533324 
		+ 1.139449335841631*logtev 
		- 0.1421013521554148*logtev2 
		+ 0.00846445538663*logtev3 
		- 0.0014327641212992*logtev4 
		+ 0.0002012250284791*logtev5 
		+ 0.0000866396324309*logtev6 
		- 0.00002585009680264*logtev7 
		+ 2.4555011970392e-6*logtev8 
		- 8.06838246118e-8*logtev9));
  }else{
    return (2.5634e-9*pow(tev,1.78186));
  }

}

double k18(double t)
{
  double tev;

  tev = t*K_to_eV;

  return (6.5e-9/sqrt(tev));

  // Galli & Palla (1998)
  // 5.7e-6/sqrt(t) + 6.3e-8 - 9.2e-11*sqrt(t) + 4.4e-13*t
}

double k19(double t)
{
  if(t>1.0e4) {
    return (4.0e-4*pow(t,-1.4)*exp(-15100.0/t));
  }else{
    return (1.0e-8*pow(t,-0.4));
  }

  // Galli & Palla (1998)
  // if (t>8.0e3) {
  //    return (9.6e-7*pow(t,-0.9));
  // }else{
  //    return (6.9e-9*pow(t,-0.35));
  // }
}

double k20(double t)
{
  // double tev;

  // tev = t*K_to_eV;

  // Lepp & Shull
  // return (5.56396e-8/pow(tev,0.6035));

  // Galli & Palla (1998)
  // return (2.0e-7/sqrt(t));

  if(t<617.0) {
    return (1.e-8);
  }else{
    return (1.32e-6*pow(t,-0.76));
  }
}

double k21(double t)
{
  double tev;

  tev = t*K_to_eV;

  return (4.64e-8/sqrt(tev));
}
