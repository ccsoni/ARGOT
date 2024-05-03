//File include by calc_rmesh_data.c //

//=======================================================================
//
//   reactions for the chemical network
//   k02    : HII   + e   -> HI    + photon
//   k04    : HeII  + e   -> HeI   + photon
//   k06    : HeIII + e   -> HeII  + photon
//
//=======================================================================


inline double k02_A(double t)
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
}


inline double k02_B(double t)
{
  /* Hui + Gnedin 1997*/ 
  
  double lambda = 2.0*157807.0/t;
  return (2.753e-14*pow(lambda,1.5)/pow(1.0+pow(lambda/2.740, 0.407),2.242));
    
}


inline double k04_A(double t) 
{
  double tev;
  
  tev = t*K_to_eV;

  if (tev > 0.8) {
    return (1.54e-9*(1.+0.3/exp(8.099328789667/tev)) 
	    / (exp(40.49664394833662/tev)*pow(tev,1.5)) 
	    + 3.92e-13/pow(tev,0.6353));
  }else{
    return 3.92e-13/pow(tev,0.6353);
  }
}


inline double k04_B(double t) 
{
  if(t>500.3) { // if log10(T) > 2.69923
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
}



inline double k06_A(double t)
{
  return (3.36e-10/sqrt(t)/pow((t/1.e3),0.2)/(1.0+pow((t/1.e6),0.7)));
}


inline double k06_B(double t)
{
  double lambda = 2.0*631515.0/t;
  return (1.3765e-14*pow(lambda,1.5)/pow(1.0+pow(lambda/2.740, 0.407),2.242));
}
