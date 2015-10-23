/*******************************************************************************
* Piotr's Image&Video Toolbox      Version 3.21
* Copyright 2013 Piotr Dollar.  [pdollar-at-caltech.edu]
* Please email me if you find bugs, or have suggestions or questions!
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#include <mex.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

typedef unsigned char uint8;
typedef unsigned int uint32;
#define min(x,y) ((x) < (y) ? (x) : (y))

template<typename T>
void forestInds( float *retdata, 
  const T *data, const T *thrs, const uint32 *fids, const uint32 *child, 
  const uint32 *node2leaf, 
  Eigen::MatrixXf &myfeats, Eigen::MatrixXf &mydata, 
  std::vector<Eigen::MatrixXf> & R2, 
  int N, int nThreads, 
  int splitfuntype, int NN, int leafpredtype, int tid, int treeoff )
{
   for( int i = 0; i < N; i++ ) {
     uint32 k = 0;
     int leafid = 0;
     while( child[k] )
     {
       if (splitfuntype==1)
       {
         if( data[i+fids[k]*N] < thrs[k] )
           k = child[k]-1; else k = child[k];
       }
       else
       {
         if( (data[i+fids[k]*N]-data[i+fids[k+NN]*N]) < thrs[k])
           k = child[k]-1; else k = child[k];
       }
     }
     //  leaf node id
     leafid = node2leaf[k]-1;
     
     // compute HR reconstruction
     Eigen::VectorXf reconstVector;
     if (leafpredtype==0) // constant
     {
       reconstVector = R2[leafid];
     }
     else if (leafpredtype==1 || leafpredtype==2) // linear model (either linear or polynomial basis functions)
     {
       Eigen::VectorXf datasample = mydata.row(i);
       reconstVector = R2[leafid] * datasample;
     }
     for (int f = 0; f < reconstVector.rows(); ++f)
       retdata[tid*treeoff + i*reconstVector.rows() + f] = reconstVector(f);
   }
}

// retdata=mexFunction(Xfeat,Xsrc,forest,node2leafids,leafs,leafpredtype,[nThreads])
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // variable declarations
  int N, nThreads, splitfuntype, Ff, Fd, treeoff, Nrows, Ncols, leafpredtype;
  void *feat, *data;
  float *retdata;
  
  // ======================================================
  // data
  feat = mxGetData(prhs[0]);
  data = mxGetData(prhs[1]);
  N = (int) mxGetM(prhs[0]);
  Ff = (int) mxGetN(prhs[0]);
  Fd = (int) mxGetN(prhs[1]);
  // map the data into Eigen matrices
  Eigen::Map<Eigen::MatrixXf> myfeatsMap((float*)feat,N,Ff);
  Eigen::Map<Eigen::MatrixXf> mydataMap((float*)data,N,Fd);
  Eigen::MatrixXf myfeats = myfeatsMap;
  Eigen::MatrixXf mydata = mydataMap;
  //std::cout << myfeats.rows() << "x" << myfeats.cols() << std::endl;
  //std::cout << mydata.rows() << "x" << mydata.cols() << std::endl;
  
  // ======================================================
  // forest
  if (!mxIsStruct(prhs[2]))
    mexErrMsgTxt("forest should be a struct");
  size_t ntrees = mxGetNumberOfElements(prhs[2]);
  //std::cout << "#trees = " << ntrees << std::endl;
  if (!mxIsCell(prhs[4]) || mxGetNumberOfElements(prhs[4])!=ntrees)
    mexErrMsgTxt("leafs should be a cell array of size #trees");
  if (ntrees < 1)
    mexErrMsgTxt("At least one tree should be given");
  mxArray *tree0leafs = mxGetCell(prhs[4],0);
  mxArray *firstT = mxGetField(tree0leafs,0,"T");
  Nrows = (int) mxGetM(firstT);
  Ncols = (int) mxGetN(firstT);
  //std::cout << "T: " << Nrows << " x " << Ncols << std::endl;
  splitfuntype =  (int) mxGetN(mxGetField(prhs[2],0,"fids")); // 1=single, 2=pair
  if (splitfuntype != 1 && splitfuntype != 2)
    mexErrMsgTxt("Splitfuntype is not correct, size(fids,2)={1,2}!.");
  //std::cout << "Splitfuntype: " << splitfuntype << std::endl;
  // get the type of leaf node prediction
  leafpredtype = (int) mxGetScalar(prhs[5]);
  //std::cout << "leafpredtype = " << leafpredtype << std::endl;

  
  // ======================================================
  // check number of threads
  nThreads = (nrhs<7) ? 100000 : (int) mxGetScalar(prhs[6]);
  nThreads = min(nThreads,omp_get_max_threads());
  //std::cout << "#threads = " << nThreads << std::endl;
  

  // ======================================================
  // prepare the output variables
  plhs[0] = mxCreateCellMatrix(ntrees,1);
  
  // Do the job
  #pragma omp parallel for num_threads(nThreads)
  for (size_t t = 0; t < ntrees; ++t)
  {
    // get mxArrays for current tree
    void *treeThrs = mxGetData(mxGetField(prhs[2],t,"thrs"));
    uint32 *treeFids = (uint32*) mxGetData(mxGetField(prhs[2],t,"fids"));
    uint32 *treeChild = (uint32*) mxGetData(mxGetField(prhs[2],t,"child"));
    int treeNN = (int) mxGetM(mxGetField(prhs[2],t,"fids")); // number of nodes for this tree!
    
    
    // get mapping from node ids to leaf ids
    uint32 *node2leaf = (uint32*) mxGetData(mxGetCell(prhs[3],t));
    // get #leafs for this tree
    mxArray *leafs = mxGetCell(prhs[4],t);
    if (!mxIsStruct(leafs))
      mexErrMsgTxt("leafs should be a struct array");
    int nleafs = (int) mxGetNumberOfElements(leafs);
    // map the leaf node matrices into vectors of Eigen matrices
    std::vector<Eigen::MatrixXf> R2(nleafs);
    int R2fieldnum = mxGetFieldNumber(leafs,"T");
    mxArray *blub; void *tmpmat;
    int NrowsTmp, NcolsTmp;
    for (size_t l = 0; l < nleafs; ++l)
    {
      // get second matrix depending on the type
      blub = mxGetFieldByNumber(leafs,l,R2fieldnum);
      NrowsTmp = (int) mxGetM(blub);
      NcolsTmp = (int) mxGetN(blub);
      tmpmat = mxGetData(blub);
      Eigen::Map<Eigen::MatrixXf> myR2((float*)tmpmat,NrowsTmp,NcolsTmp);
      R2[l]=myR2;
    }
    
    mxArray* treeRetData = mxCreateNumericMatrix(Nrows,N,mxSINGLE_CLASS,mxREAL);
    retdata = (float*) mxGetPr(treeRetData);
    mxSetCell(plhs[0],t,treeRetData);
    treeoff = 0;
    
    // call the random forest
    if(mxGetClassID(prhs[0])!=mxGetClassID(mxGetField(prhs[2],t,"thrs"))) // compare data with tree.thrs
      mexErrMsgTxt("Mismatch between data types.");
    if(mxGetClassID(prhs[0])==mxSINGLE_CLASS)
      forestInds(retdata,(float*)feat,(float*)treeThrs,treeFids,treeChild,node2leaf,myfeats,mydata,R2,N,nThreads,splitfuntype,treeNN,leafpredtype,t,treeoff);
    else if(mxGetClassID(prhs[0])==mxDOUBLE_CLASS)
      forestInds(retdata,(double*)feat,(double*)treeThrs,treeFids,treeChild,node2leaf,myfeats,mydata,R2,N,nThreads,splitfuntype,treeNN,leafpredtype,t,treeoff);
    else if(mxGetClassID(prhs[0])==mxUINT8_CLASS)
      forestInds(retdata,(uint8*)feat,(uint8*)treeThrs,treeFids,treeChild,node2leaf,myfeats,mydata,R2,N,nThreads,splitfuntype,treeNN,leafpredtype,t,treeoff);
    else mexErrMsgTxt("Unknown data type.");
  }

  return;
}
