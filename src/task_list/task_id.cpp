//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
//! \file task_id.cpp
//  \brief implementation of the TaskID class

#include <iostream>
#include "tasks.hpp"

#define DEBUG_TASKID 0

// TaskID constructor. Default id = 0.

TaskID::TaskID(unsigned int id) {
  if (id > 0) bitfld_.set(id-1);
}

void TaskID::Print(const std::string label) {
  std::cout << label << " " << bitfld_.to_string() << std::endl;
}

//----------------------------------------------------------------------------------------
//! \fn void TaskID::Clear()
//  \brief Clear all the bits in the TaskID

void TaskID::clear() {
  bitfld_.reset();
}

//----------------------------------------------------------------------------------------
//! \fn bool TaskID::IsUnfinished(const TaskID& id)
//  \brief Check if the task with the given ID is unfinished. This function is to be
//  called on Task States and returns true if the task is unfinished.

bool TaskID::IsUnfinished(const TaskID& id) const {
#if DEBUG_TASKID
  std::cout << "IsUnfinished " << bitfld_.to_string() << std::endl
            << "             " << id.bitfld_.to_string() << std::endl
            << "             " << (bitfld_ & id.bitfld_).to_string()
            << "             " << (bitfld_ & id.bitfld_).none()
            << std::endl << std::endl;
#endif
  return (bitfld_ & id.bitfld_).none();
}

//----------------------------------------------------------------------------------------
//! \fn bool TaskID::CheckDependencies(const TaskID& dep)
//  \brief Check if the given dependencies are cleared. This function is to be
//  called on Task States, and returns true if all the dependencies are clear.

bool TaskID::CheckDependencies(const TaskID& dep) const {
#if DEBUG_TASKID
  std::cout << "CheckDepende " << bitfld_.to_string() << std::endl
            << "        dep  " << dep.bitfld_.to_string() << std::endl
            << "       dep2  " << (bitfld_ & dep.bitfld_).to_string() << std::endl
	    << "     depAll  " << (bitfld_ & dep.bitfld_).all()       << std::endl
            << std::endl << std::endl;
#endif
  return ((bitfld_ & dep.bitfld_) == dep.bitfld_);
}


//----------------------------------------------------------------------------------------
//! \fn void TaskID::SetFinished(const TaskID& id)
//  \brief Mark the task with the given ID finished.
//  This function is to be called on Task States.

void TaskID::SetFinished(const TaskID& id) {
  #if DEBUG_TASKID
  std::cout << "SetFinished  " << bitfld_.to_string() << std::endl
            << "             " << id.bitfld_.to_string() << std::endl
            << "             " << (bitfld_ ^ id.bitfld_).to_string() << std::endl << std::endl;
  #endif
  bitfld_ ^= id.bitfld_;
}


//----------------------------------------------------------------------------------------
//! \fn bool TaskID::operator== (const TaskID& rhs)
//  \brief overloading operator == for TaskID

bool TaskID::operator== (const TaskID& rhs) const {
  return (bitfld_ == rhs.bitfld_);
}

//----------------------------------------------------------------------------------------
//! \fn TaskID TaskID::operator| (const TaskID& rhs)
//  \brief overloading operator | for TaskID

TaskID TaskID::operator| (const TaskID& rhs) const {
  TaskID ret;
  ret.bitfld_ = (bitfld_ | rhs.bitfld_);
  return ret;
}
