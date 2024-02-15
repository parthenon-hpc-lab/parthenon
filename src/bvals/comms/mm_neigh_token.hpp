#ifndef BVALS_COMMS_MM_NEIGH_TOKEN_HPP_
#define BVALS_COMMS_MM_NEIGH_TOKEN_HPP_

#define USE_NEIGHBORHOOD_COLLECTIVES
#include <parthenon_mpi.hpp>
#include <utils/error_checking.hpp>
#include <stdlib.h> 
#include "globals.hpp"
#include "kokkos_abstraction.hpp"
#include "basic_types.hpp"

namespace neigh_comm{
    class NeighToken{
        using p_key = std::tuple<int, int, std::string, int>;
        using buff_sizes = std::vector<std::pair<p_key,size_t>>;

        public:
            
            // , send_comm_buffer(("send_comm_buffer",100)), recv_comm_buffer(("recv_comm_buffer",100))
            
            /*
             * add_buff_info()
             */
            void add_buff_info(int neigh_mpi_rank, const int & buff_size){
                if(neigh_mpi_rank >= 0 ){
                    if(building_token_on){
                        mpi_neighbors.insert(neigh_mpi_rank);
                        total_buff_size_per_rank[neigh_mpi_rank] += buff_size;
                        total_buf_size += buff_size;
                    }
                }
                else{
                    PARTHENON_FAIL("trying to add a negative mpi rank in NeighToken::add_neighbor (neigh_mpi_rank < 0)");
                }
            }

            /*
             * start_building_buffers()
             */
            void start_building_buffers(){
                if(building_token_on)  PARTHENON_FAIL("starting illegal building buffers");
                building_token_on = true;

                mpi_neighbors.clear();
                displs.clear();
                offsets.clear();
                counts.clear();
                total_buff_size_per_rank.clear();
                total_buf_size=0;
            }

            /*
             * end_building_buffers()
             */
            void end_building_buffers(){
                building_token_on=false;
            }

            /*
             * build_neigh_comm()
             */
            void build_neigh_comm(MPI_Comm comm_){
                std::vector<int> mpi_procs;
                for(int rank : mpi_neighbors){
                    mpi_procs.push_back(rank);
                }
                
                // create the neigh communicator
                MPI_Dist_graph_create_adjacent(comm_, mpi_procs.size(), mpi_procs.data(), MPI_UNWEIGHTED,
                                   mpi_procs.size(), mpi_procs.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, false, &neigh_comm);
                
            }
            /*
             * calculate_off_prefix_sum()
             * calculate offsets
             */
            void calculate_off_prefix_sum(){
                if(building_token_on){
                    displs.reserve(mpi_neighbors.size());
                    displs.push_back(0);
                    int indx = 1;
                    // compute the displacements and offsets
                    for(int neigh : mpi_neighbors){
                        int buff_size_curr_neigh = total_buff_size_per_rank[neigh];
                        if(indx < mpi_neighbors.size())
                            displs.push_back(buff_size_curr_neigh+displs[indx - 1]);
                        offsets[neigh] = displs[indx - 1];
                        counts.push_back(buff_size_curr_neigh);
                        indx++;
                    }
                }
            }

            /*
             * alloc_comm_buffers()
             */
            void alloc_comm_buffers(){
                if(building_token_on){
                    /*if(send_comm_buffer.extent(0) < total_buf_size)
                        new(&send_comm_buffer) Kokkos::View<parthenon::Real*>("send_comm_buffer", total_buf_size);

                    if(recv_comm_buffer.extent(0) < total_buf_size)
                        new(&recv_comm_buffer) Kokkos::View<parthenon::Real*>("recv_comm_buffer", total_buf_size);*/

                    if(send_comm_buffer.extent(0) < total_buf_size)
                        realloc(send_comm_buffer, total_buf_size);

                    if(recv_comm_buffer.extent(0) < total_buf_size)
                        realloc(recv_comm_buffer, total_buf_size);
                }
            } 

            /*
             * start_data_exchange_neigh_alltoallv()
             */
            void start_data_exchange_neigh_alltoallv(){
                MPI_Ineighbor_alltoallv(send_comm_buffer.data(), counts.data(), displs.data(), MPI_PARTHENON_REAL, 
                recv_comm_buffer.data(), counts.data(), displs.data(), MPI_PARTHENON_REAL, neigh_comm, &neigh_request);
            }
            
            /*
             * end_data_exchange_neigh_alltoallv()
             */
            bool end_data_exchange_neigh_alltoallv(){
                int flag = 0;
                if(neigh_request)
                    MPI_Test(&neigh_request,&flag, MPI_STATUS_IGNORE);
                return flag;
            }

        public:
            NeighToken(): building_token_on(false), neigh_request(), send_comm_buffer("send_neigh_buf",100), recv_comm_buffer("recv_neigh_buf",100) {}
            
            std::set<int> mpi_neighbors;
            std::vector<int> displs;
            std::vector<int> counts;
            std::map<int, int> offsets;
            std::map<int, int> total_buff_size_per_rank;
            int total_buf_size;

            MPI_Request neigh_request;

            bool building_token_on;
            bool enable_add_buff;
            MPI_Comm neigh_comm; // created with MPI_Dist_graph_create_adjacent

            //Kokkos::View<parthenon::Real*> 
            Kokkos::View<parthenon::Real*, parthenon::LayoutWrapper, parthenon::BufMemSpace> send_comm_buffer;
            //Kokkos::View<parthenon::Real*> 
            Kokkos::View<parthenon::Real*, parthenon::LayoutWrapper, parthenon::BufMemSpace> recv_comm_buffer;

    };

}
#endif