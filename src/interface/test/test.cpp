//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
// g++ -Wall --std=c++11 -I../src -I. test.cpp -o test && ./test

#include "test.hpp"

using namespace parthenon;
int main() {
  //  testMetadata();
  //  testVariable();
  testContainer();
  return 0;
}

void testContainer() {
  MeshBlock mb(10,20,30);
  std::cout << "\n\n______________TESTING CONTAINER_______________" << std::endl;

  Container<double> c;  // our container!!!
  std::vector<int> three {20,10,5};
  std::vector<int> two {20,5};
  std::vector<int> one {7};

  Metadata m; // metadata structure

  // set the block size for the container
  c.setBlock(&mb);

  // set number of materials for simulation
  c.setNumMat(3);

  m.reset();  // not needed here because m starts empty, but in case we relocate code
  m.setMultiple({m.face, m.intensive, m.oneCopy}); // at this point m flags are face,intensive,oneCopy

  std::cout << "_____Test Add_______" << std::endl;
  c.Add(std::string("Face_first"), m, three);  // add a threeD intensive face variable
  c.Add(std::string("Face_two"), m, two);      // add a twoD extensive face variable

  m.set(m.advected); // at this point m flags are face,intensive,oneCopy,advected
  c.Add(std::string("Face_Advected"), m, two);
  {
    FaceField *d, *e;
    std::cout << "_____Test Get_______" << std::endl;
    d = c.GetFace("Face_two");
    std::cout << "    -- 'error' --" << std::endl;
    try {
      e = c.GetFace("iShouldFail");
    }
    catch (const std::invalid_argument& x) {
      std::cout << "  --Get failed for 'iShouldFail' as expected with: " << x.what() << std::endl;
    }
  }

  // reset the metadata and give new flags for new variables
  m.reset();  // at this point flags are null
  m.setMultiple({m.cell, m.intensive, m.oneCopy});  // at this point m flags are cell,intensive,oneCopy
  c.Add(std::string("Cell"), m, two);
  m.set(m.advected);
  m.setMaterials(true);
  c.Add(std::string("Cell_Advected"), m, two);
  c.Add(std::string("Cell 0D"), m);
  c.Add(std::string("Cell 1D"), m, one);
  m.setMaterials(false);

  // print out the container variables
  c.print();

  std::cout << std::endl;
  std::cout << "================= TESTING ITERATORS ==================\n\n";

  ContainerIterator<double> ci(c,{m.face});
  std::cout << "=================================\n";
  std::cout << "      FACE VARIABLES\n";
  std::cout << "=================================\n";
  for (auto &v : ci.varsFace) {
    std::cout << "______________________ITER VAR="<<v.info()<<std::endl;
  }

  ci.setMask({m.materials});
  std::cout << "=================================\n";
  std::cout << "      MATERIAL VARIABLES\n";
  std::cout << "=================================\n";
  for (auto &v : ci.vars) {
    for (int i=0; i<3; i++) {
      // set the first entry of the first three materials to 5
      v(0,i,0,0,0) = 5;
      Real *data = v.data();
      std::cout << "______________________ITER VAR="<<v.info()<<std::endl;
      for (int i = 0; i< v.GetSize(); i++) data[i] = 7;
      std::cout << "                           MATL:" <<i << " : " << v(0,i,0,0,0) << std::endl;
    }
  }
  for (auto &v : ci.varsFace) {
    std::cout << "______________________ITER VAR="<<v.info()<<std::endl;
  }

  {
    // Testing container slice
    std::cout << "_____Test Material slice_______" << std::endl;
    Container<Real> cSlice = c.materialSlice(1);
    cSlice.print();
    ContainerIterator<double> ciSlice(cSlice,{m.materials});
    for (auto &v : ciSlice.vars) {
      Real *data = v.data();
      std::cout << "______________________SLICE VAR="<<v.info() << std::endl;
      for (int i = 0; i< v.GetSize(); i++) data[i] = -1;
    }
    std::cout << "=================================\n";
    std::cout << "  POST SLICE MATERIAL VARIABLES\n";
    std::cout << "=================================\n";
    for (auto &v : ci.vars) {
      std::cout << "_______ITER VAR="<<v.info()<<"  size:"<<v.GetSize()<<std::endl;
      for (int i=0; i<3; i++) {
	std::cout << "            MATL: " <<i << " : " ;
	switch(v.matIndex()) {
	  case 3:
	    std::cout << v(i,0,0,0);
	    std::cout << " " << v(i,1,0,0);
	    break;
	  case 4:
	    std::cout << v(i,0,0,0,0);
	    std::cout << " " << v(i,1,0,0,0);
	    break;
	  case 5:
	    std::cout << v(i,0,0,0,0,0);
	    std::cout << " " << v(i,1,0,0,0,0);
	    break;
	}
	std::cout << std::endl;
      }
    }
  }

  std::cout << std::endl;
  //  std::cout << "================================================\n\n";

  std::cout << "=================================\n";
  std::cout << "     ADVECTED FACE VARIABLES\n";
  std::cout << "=================================\n";
  ci.setMask({m.face, m.advected});
  for (auto &v : ci.varsFace) {
    std::cout << "______________________VAR="<<v.info()<<std::endl;
  }
  std::cout << std::endl;
  //  std::cout << "================================================\n\n";

  std::cout << "=================================\n";
  std::cout << "    ALL ADVECTED VARIABLES\n";
  std::cout << "=================================\n";
  ci.setMask({m.advected});
  for (auto &v : ci.vars) {
    std::cout << "______________________VAR="<<v.info()<<std::endl;
  }
  for (auto &v : ci.varsFace) {
    std::cout << "______________________VAR="<<v.info()<<std::endl;
  }
  for (auto &v : ci.varsEdge) {
    std::cout << "______________________VAR="<<v.info()<<std::endl;
  }
  std::cout << std::endl;
  //  std::cout << "================================================\n\n";

  std::cout << "=================================\n";
  std::cout << "  REDO ALL ADVECTED VARIABLES\n";
  std::cout << "=================================\n";
  for (auto &v : ci.vars) {
    std::cout << "______________________VAR="<<v.info()<<std::endl;
  }
  for (auto &v : ci.varsFace) {
    std::cout << "______________________VAR="<<v.info()<<std::endl;
  }
  for (auto &v : ci.varsEdge) {
    std::cout << "______________________VAR="<<v.info()<<std::endl;
  }
  std::cout << std::endl;
  //  std::cout << "================================================\n\n";


  // these should pass
  std::cout << "_____Test Remove_______" << std::endl;
  try {
    // this should fail
    c.Remove("iShouldFail");
    std::cout << "_________ERROR Unknown name didn't throw exception" << std::endl;
  }
  catch (const std::invalid_argument& e) {
    std::cout << "  --Remove failed for 'iShouldFail' as expected with: " << e.what() << std::endl;
  }
  c.Remove("Face_first");
  std::cout << "  -- "<< c.size() << " Remove Face_first gave success\n";
  std::cout << "  -- "<< c.size() << " is original size\n";
  c.Remove("Face_two");
  std::cout << "  -- "<< c.size() << " Remove Face_two gave success\n";

}
void testVariable() {
  std::cout << "\n\n______________TESTING VARIABLE________________" << std::endl;
  Metadata m;
  m.reset();
  m.setMultiple({m.face, m.intensive, m.oneCopy});

  std::array<int, 2> dims {100,300};
  std::cout << dims[0] << std::endl;

  std::cout << typeid(dims[0]).name() << std::endl;

  //----------------------------------------------------
  // instantiate variable of type int
  //----------------------------------------------------
  Variable<int> k = Variable<int>("myvar",dims,m);
  std::cout <<"  Dims of " << k.label() << ": " << k.GetDim2() << "," << k.GetDim1() << std::endl;

  //----------------------------------------------------
  //resize
  //----------------------------------------------------
  k.NewAthenaArray(10,20);
  std::cout <<"  New Dims of " << k.label() << ": " << k.GetDim2() << "," << k.GetDim1() << std::endl;

  //----------------------------------------------------
  //initialize some values
  //----------------------------------------------------
  for (int j=0; j<k.GetDim2(); j++) {
    for (int i=0; i<k.GetDim1(); i++) {
      k(j,i) = j*1000 + i;
    }
  }

  //----------------------------------------------------
  //get a 1D slice
  //----------------------------------------------------
  {
    Variable<int> slice1D = Variable<int>("slice1D",k,1,5,5);
    std::cout <<"  Dims of " << slice1D.info() << std::endl;

    //----------------------------------------------------
    // We expect 5-9 here because the slice1D gets us k(0,5:9) which
    //----------------------------------------------------
    int errCount = 0;
    for (int i=0; i<slice1D.GetDim1(); i++) {
      int e = i+5;
      if (slice1D(i) - e ) {
	std::cout << i << ":" << slice1D(i) << "(expected "<<e<<")"<<"  ERR=" << slice1D(i) - e<< std::endl;
	errCount++;
      }
    }
    std::cout << "__________1D slice Found " << errCount << " errors" << std::endl;
  }


  //----------------------------------------------------
  //get a 2D slice gets us k(5:9,0:9)
  //----------------------------------------------------
  {
    Variable<int> slice2D = Variable<int>("slice2D",k,2,5,5);
    std::cout <<"  Dims of " << slice2D.info() << std::endl;
    std::cout <<"  Dims of " << k.info() << std::endl;

    //----------------------------------------------------
    // Note that we expect (j+5)*1000 + i
    //----------------------------------------------------
    int errCount = 0;
    for (int j = 0; j<slice2D.GetDim2(); j++) {
      for (int i=0; i<slice2D.GetDim1(); i++) {
	int e = (j+5)*1000 + i;
	if (slice2D(j,i) - e ) {
	  std::cout << j<<","<<i << ":" << slice2D(j,i) << "(expected "<<e<<")"<<"  ERR=" << slice2D(j,i) - e<< std::endl;
	  errCount++;
	}
      }
    }
    std::cout << "__________2D slice Found " << errCount << " errors" << std::endl;
  }
  return;
}


void testMetadata() {
  std::cout << "\n\n______________TESTING METADATA________________" << std::endl;
  // tests metadata structure and returns a valid one
  Metadata m;

  //--------------
  // test topology
  //--------------
  {
    // this should throw exception
    try {
      m.setWhere(m.intensive);
    }
    catch (const std::invalid_argument& e) {
      std::cout << "  --where: exception for m.intensive\n           " << e.what() << std::endl;
    }

    // these should pass
    m.setWhere(m.none);std::cout << "  --where=" << m.flag_labels(m.where()) << std::endl;
    m.setWhere(m.cell);std::cout << "  --where=" << m.flag_labels(m.where()) << std::endl;
    m.setWhere(m.face);std::cout << "  --where=" << m.flag_labels(m.where()) << std::endl;
    m.setWhere(m.edge);std::cout << "  --where=" << m.flag_labels(m.where()) << std::endl;
    m.setWhere(m.node);std::cout << "  --where=" << m.flag_labels(m.where()) << std::endl;
  }
  unsigned long mask = m.mask();
  std::cout << "mask = " << mask << std::endl;

  return;
}
