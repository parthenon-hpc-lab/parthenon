#ifndef UTILS_MM_LOGGER_HPP_
#define UTILS_MM_LOGGER_HPP_
#include <mpi.h>
#include <fstream>
#include <cstdlib>

//#define ENABLE_MM_LOGGER

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
                        time_token_creation(0), token_id(0)  {
                const char * env_filename = getenv("MM_LOGGER_OUT_FILE");
                if(env_filename != NULL) filename = env_filename;
                _start_timer(total_exec_time);
            }

            ~My_Logger() { 
                if(is_init){
                    _end_timer(total_exec_time);
                    log_stream<<"#END:"<<total_exec_time<<std::endl;
                    log_file << log_stream.rdbuf();
                    log_file.close();
                }
            }

            void init_logger( const int & rank){
                if(!is_init){
                    filename = filename + "_" + std::to_string(rank);
                    log_file.open(filename);
                    is_init=true;
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

            void _start_timer(double & t){t -= MPI_Wtime();}
            void _end_timer(double & t){t += MPI_Wtime();}
    };
    #ifdef ENABLE_MM_LOGGER
    extern std::shared_ptr<My_Logger> global_logger;
    #endif
}
#endif