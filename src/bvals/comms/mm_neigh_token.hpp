#ifndef BVALS_COMMS_MM_NEIGH_TOKEN_HPP_
#define BVALS_COMMS_MM_NEIGH_TOKEN_HPP_

//#define USE_NEIGHBORHOOD_COLLECTIVES
#include <parthenon_mpi.hpp>
#include <utils/error_checking.hpp>
#include <stdlib.h> 
#include <globals.hpp>
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
            void add_buff_info(int neigh_mpi_rank, const int buff_size, const int tag){
                if(neigh_mpi_rank >= 0 ){
                    if(building_token_on){
                        mpi_neighbors.insert(neigh_mpi_rank);
                        total_buff_size_per_rank[neigh_mpi_rank] += buff_size;
                        total_buf_size += buff_size;
                        buff_info_per_rank[neigh_mpi_rank].push_back({tag,buff_size});
                    }
                }
                else
                    PARTHENON_FAIL("trying to add a negative mpi rank in NeighToken::add_neighbor (neigh_mpi_rank < 0)");
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
                buff_info_per_rank.clear();
                per_tag_offsets.clear();
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
                int tmp_rank;
                MPI_Comm_rank(MPI_COMM_WORLD,&tmp_rank);
                for(int rank : mpi_neighbors){
                    mpi_procs.push_back(rank);
                }
                // create the neigh communicator
                MPI_Dist_graph_create_adjacent(comm_, mpi_procs.size(), mpi_procs.data(), MPI_UNWEIGHTED,
                                   mpi_procs.size(), mpi_procs.data(), MPI_UNWEIGHTED,
                                   MPI_INFO_NULL, false, &neigh_comm);
                nb_of_comm_built++;
            }
            /*
             * calculate_off_prefix_sum()
             * calculate offsets
             */
            void calculate_off_prefix_sum(bool debug_info=false){
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

                    // sort buffer info for each rank
                    for(auto & [neigh, info_v] : buff_info_per_rank){
                        std::sort(info_v.begin(), info_v.end(), [](auto &left, auto &right) {
                            return left.first < right.first;
                        });

                        int global_offset = offsets[neigh];
                        int curr_offset = global_offset;
                        for(auto& [tag, buff_size] : info_v){ // sorted tags
                            int end_offset = curr_offset + buff_size - 1;
                            per_tag_offsets[neigh][tag] = {curr_offset,end_offset};
                            curr_offset += buff_size;
                        }
                    }

                    /*if(debug_info){
                        std::cout<<std::endl<<std::endl;
                        for(auto & [neigh, info_v] : buff_info_per_rank){
                            std::cout<<"["<<neigh<<"]"<<"orig offset : "<<offsets[neigh]<<std::endl;
                            std::cout<<"["<<neigh<<"]"<<"new offsets :";
                            for(auto& [tag, buff_size] : info_v){
                                std::cout<<"("<<tag<<","<<per_tag_offsets[neigh][tag].first<<
                                ","<<per_tag_offsets[neigh][tag].second<<"),";
                            }
                            std::cout<<std::endl;
                        }
                        std::cout<<std::endl<<std::endl;
                    }*/

                }
            }

            /*
             * alloc_comm_buffers()
             */
            void alloc_comm_buffers(){
                if(building_token_on){
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
                if(neigh_comm_in_use) PARTHENON_FAIL("trying to launch an alltoallv operation while the communicator is in use");
                neigh_comm_in_use = true;
                MPI_Ineighbor_alltoallv(send_comm_buffer.data(), counts.data(), displs.data(), MPI_PARTHENON_REAL, 
                recv_comm_buffer.data(), counts.data(), displs.data(), MPI_PARTHENON_REAL, neigh_comm, &neigh_request);
            }
            
            /*
             * test_data_exchange_neigh_alltoallv()
             */
            bool test_data_exchange_neigh_alltoallv(){
                int flag_nc = 0;
                //if(neigh_request)
                MPI_Test(&neigh_request, &flag_nc, MPI_STATUS_IGNORE);
                if(flag_nc) neigh_comm_in_use = false;
                //MPI_Wait(&neigh_request, MPI_STATUS_IGNORE);
                //return true;
                return flag_nc;
            }

            ~NeighToken(){
                int root_rank = 0;
                if(parthenon::Globals::my_rank == root_rank)
                    std::cout<<"# COMM_BUILD_INFO: Nb_of_comm_build="<<nb_of_comm_built<<std::endl;
            }

        public:
            NeighToken(): building_token_on(false), neigh_request(), nb_of_comm_built(0), send_comm_buffer("send_neigh_buf",100), recv_comm_buffer("recv_neigh_buf",100), neigh_comm_in_use(false) {}
            
            std::set<int> mpi_neighbors;
            std::vector<int> displs;
            std::vector<int> counts;
            std::map<int, int> offsets;
            std::map<int, std::map<int, std::pair<int,int>>> per_tag_offsets;
            std::map<int, int> total_buff_size_per_rank;
            std::map<int, std::vector<std::pair<int,int>>> buff_info_per_rank;
            int total_buf_size;
            size_t nb_of_comm_built;

            MPI_Request neigh_request;

            bool building_token_on;
            bool enable_add_buff;
            bool neigh_comm_in_use;
            MPI_Comm neigh_comm; // created with MPI_Dist_graph_create_adjacent

            //Kokkos::View<parthenon::Real*> 
            Kokkos::View<parthenon::Real*, parthenon::LayoutWrapper, parthenon::BufMemSpace> send_comm_buffer;
            //Kokkos::View<parthenon::Real*> 
            Kokkos::View<parthenon::Real*, parthenon::LayoutWrapper, parthenon::BufMemSpace> recv_comm_buffer;

    };
}
#endif