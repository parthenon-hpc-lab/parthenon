//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

// This file was made in part with generative AI.

#include <format>
#include <fstream>
#include <string>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "utils/calc_spectrum.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void SpectralOutput::WriteOutputFile()
//  \brief Writes a spectrum output file

void SpectralOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                     const SignalHandler::OutputSignal signal) {
  const auto var_name = pin->GetString(output_params.block_name, "variable");
  const auto components = pin->GetVector<int>(output_params.block_name, "components");
  const auto output_label =
      pin->GetOrAddString(output_params.block_name, "output_label", var_name);

  auto spectra = CalcSpectrum(pm, var_name, components);
  auto spectra_h = spectra.GetHostMirrorAndCopy();
  const auto num_bins = spectra_h.extent(0);

  if (parthenon::Globals::my_rank == 0) {
    std::string suffix;
    if (signal == SignalHandler::OutputSignal::now) {
      suffix = "now";
    } else if (signal == SignalHandler::OutputSignal::final &&
               output_params.file_label_final) {
      suffix = "final";
    } else {
      suffix = std::format("{:0{}d}", output_params.file_number,
                           output_params.file_number_width);
    }

    const std::string fname = std::format("{}.{}.{}.{}.spc", output_params.file_basename,
                                          output_label, output_params.file_id, suffix);

    std::ofstream fout(fname);
    if (!fout.is_open()) {
      PARTHENON_FAIL("Could not open " + fname + " for writing");
    }

    fout << "# Bin    val_sum    K_sum    Count\n";
    for (int i = 0; i < static_cast<int>(num_bins); ++i) {
      fout << std::format("{:d} {:.15e} {:.15e} {:.15e}\n", i, spectra_h(i, 0),
                          spectra_h(i, 1), spectra_h(i, 2));
    }
    fout.close();
  }

  UpdateNextOutput_(pm, tm);
}

} // namespace parthenon
