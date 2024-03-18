#ifndef UTILS_MM_LOGGER_HPP_
#define UTILS_MM_LOGGER_HPP_
#include <parthenon_mpi.hpp>
#include <fstream>
#include <cstdlib>
#include <globals.hpp>

//#define ENABLE_MM_LOGGER
//#define ENABLE_MM_LOG_TIME
namespace logger{
    
    // COMM_TYPE
    enum COMM_TYPE 
    {   
        NoneComeType = 0,
        BoundBufs = 1, 
        FluxCorrections = 2
    };

    // M_Logger
    class My_Logger{
        public:
            My_Logger(): filename("mm_logger.log"), is_init(false), time_recv_bound_bufs(0),\
                        time_send_bound_bufs(0), time_recv_flux_corr(0), time_send_flux_corr(0),\
                        time_token_creation(0), token_id(0), print_only(false), log_times(false), rank(-1)  {
                const char * env_filename = getenv("MM_LOGGER_OUT_FILE");
                if(env_filename != NULL) filename = env_filename;
                //_start_timer(total_exec_time);
            }

            #ifdef ENABLE_MM_LOG_TIME
            /* Only used for logging time info */
            My_Logger(bool _print_only): filename("mm_logger.log"), is_init(false), time_recv_bound_bufs(0),\
                        time_send_bound_bufs(0), time_recv_flux_corr(0), time_send_flux_corr(0),\
                        time_token_creation(0), token_id(0), print_only(_print_only), log_times(true),\
                        log_time_sends(0), log_time_recvs(0), log_time_build_comm(0), rank(-1)   {
                const char * env_filename = getenv("MM_LOGGER_OUT_FILE");
                if(!print_only && env_filename != NULL) filename = env_filename;
                
            }

            /* Sends */
            void start_timer_sends(){_start_timer(log_time_sends);}
            void end_timer_sends(){_end_timer(log_time_sends);}

            /* Recv */
            void start_timer_recvs(){_start_timer(log_time_recvs);}
            void end_timer_recvs(){_end_timer(log_time_recvs);}

            /* Build communicaiton token */
            void start_timer_build_comm(){_start_timer(log_time_build_comm);}
            void end_timer_build_comm(){_end_timer(log_time_build_comm);}

            #endif

            ~My_Logger() { }

            void init_logger( const int & _rank){
                if(!is_init){
                    rank = _rank;
                    if(!print_only){
                        filename = filename + "_" + std::to_string(rank);
                        log_file.open(filename);
                    }
                    is_init=true;
                    _start_timer(total_exec_time);
                }
            }

            void end_logger(){
                if(is_init){
                    _end_timer(total_exec_time);
                    #ifdef ENABLE_MM_LOG_TIME
                    double compute_time = total_exec_time - log_time_recvs - log_time_sends - log_time_build_comm;
                    #endif 
                    if(!print_only){
                        log_stream<<"#END:"<<total_exec_time<<std::endl;
                        #ifdef ENABLE_MM_LOG_TIME
                        log_stream<<"# RANK "<<rank<<" t_exec_time="<<total_exec_time
                        << " sends="<<log_time_sends<<" recvs="<<log_time_recvs
                        <<" comm_build="<<log_time_build_comm<<" compute="<<compute_time<<std::endl;
                        #endif
                        log_file << log_stream.rdbuf();
                        log_file.close();
                    }
                    #ifdef ENABLE_MM_LOG_TIME
                    else{
                        double avg_exec_time=0.0, avg_time_recvs, avg_time_build_comm, avg_compute_time; // avg_time_sends (sends are always 0 in our case)
                        int root_rank = 0;

                        // avg exec time
                        MPI_Reduce(&total_exec_time, &avg_exec_time, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
                        if(parthenon::Globals::my_rank == root_rank) avg_exec_time /= parthenon::Globals::nranks;

                        // avg recvs time
                        MPI_Reduce(&log_time_recvs, &avg_time_recvs, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
                        if(parthenon::Globals::my_rank == root_rank) avg_time_recvs /= parthenon::Globals::nranks;

                        // avg build comm time
                        MPI_Reduce(&log_time_build_comm, &avg_time_build_comm, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
                        if(parthenon::Globals::my_rank == root_rank) avg_time_build_comm /= parthenon::Globals::nranks;

                        // avg compute time
                        MPI_Reduce(&compute_time, &avg_compute_time, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);
                        if(parthenon::Globals::my_rank == root_rank) avg_compute_time /= parthenon::Globals::nranks;

                        if(parthenon::Globals::my_rank == root_rank){
                            std::cout<<"# AVG_TIMES: avg_exec_time="<<avg_exec_time
                           <<" recvs="<<avg_time_recvs<<" comm_build="<<avg_time_build_comm
                           <<" compute="<<avg_compute_time<<std::endl;
                        }
                        if(false){
                            std::cout<<"# RANK "<<rank<<" t_exec_time="<<total_exec_time
                            << " sends="<<log_time_sends<<" recvs="<<log_time_recvs
                            <<" comm_build="<<log_time_build_comm<<" compute="<<compute_time<<std::endl;
                        }
                    }
                    #endif
                }
            }

            std::stringstream& get_logger(){return log_stream;}
            
            /*std::ostream& operator<<(ostream& os, const std::string & collected_info)
            {
                log_stream << collected_info;
                return log_stream;
            }*/

            /* Timer functions */
            void start_timer_recv_bound_bufs(){_start_timer(time_recv_bound_bufs);}
            void end_timer_recv_bound_bufs(){_end_timer(time_recv_bound_bufs);}
            void log_time_recv_bound_bufs(){
                log_stream<<"HF:"<<time_recv_bound_bufs<<std::endl; // halo from (recv)
                time_recv_bound_bufs = 0;
            } 

            void start_timer_send_bound_bufs(){_start_timer(time_send_bound_bufs);}
            void end_timer_send_bound_bufs(){_end_timer(time_send_bound_bufs);}
            void log_time_send_bound_bufs(){
                log_stream<<"HT:"<<time_send_bound_bufs<<std::endl; // halo to (send)
                time_send_bound_bufs = 0;
            } 
            
            void start_timer_recv_flux_corr(){_start_timer(time_recv_flux_corr);}
            void end_timer_recv_flux_corr(){_end_timer(time_recv_flux_corr);}
            void log_time_recv_flux_corr(){
                log_stream<<"FF:"<<time_recv_flux_corr<<std::endl; // flux correction from (recv)
                time_recv_flux_corr = 0;
            } 

            void start_timer_send_flux_corr(){_start_timer(time_send_flux_corr);}
            void end_timer_send_flux_corr(){_end_timer(time_send_flux_corr);}
            void log_time_send_flux_corr(){
                log_stream<<"FT:"<<time_send_flux_corr<<std::endl;
                time_send_flux_corr = 0; // flux correction to (send)
            } 

            void start_timer_token_creation(){_start_timer(time_token_creation);}
            void end_timer_token_creation(){_end_timer(time_token_creation);}
            void log_time_token_creation(){
                log_stream<<"#TC:"<< token_id <<":"<<time_token_creation<<std::endl;
                time_token_creation = 0; 
                token_id++;
            } 
            bool new_token_created(){return (time_token_creation == 0) ? false : true;}

        private:
            bool is_init;
            bool print_only; // used only when ENABLE_MM_LOG_TIME is on
            bool log_times;  // used only when ENABLE_MM_LOG_TIME is on
            double log_time_sends;
            double log_time_recvs;
            double log_time_build_comm;

            std::string filename;
            std::stringstream log_stream;
            std::ofstream log_file;

            double time_recv_bound_bufs;
            double time_send_bound_bufs;
            double time_recv_flux_corr;
            double time_send_flux_corr;

            double time_token_creation;
            double total_exec_time;
            int token_id;

            int rank;

            void _start_timer(double & t){t -= MPI_Wtime();}
            void _end_timer(double & t){t += MPI_Wtime();}
    };
    #if defined(ENABLE_MM_LOGGER) || defined(ENABLE_MM_LOG_TIME)
    extern std::shared_ptr<My_Logger> global_logger;
    #endif
}
#endif