#pragma once

struct full_time {
	int year;    // 年，如 2025
	int month;   // 月，1-12
	int day;     // 日，1-31
	int hour;    // 时，0-23
	int minute;  // 分，0-59
	int second;  // 秒，0-59
	int usec;    // 微秒，0-999999
};

//filterbank header
struct filerbank_header {
	int Machine_id;
	int Telescope_id;
	int Data_type;
	int Nchans;
	int Nbits;
	int Nifs;
	int Barycentric;
	int Pulsarcentric;
	double Tstart;
	double Tsamp;
	double Fch1;
	double Foff;
	double RefDM;
	double Az_start;
	double Za_start;
	double Src_raj;
	double Src_dej;
	int Nbeams;
	int Ibeam;
	char Source_name[80];
};

struct __attribute__((packed)) filerbank_header_buffer {
	int HEADER_START_len;
	char HEADER_START[12];
	int source_name_len;
	char source_name[11];
	int source_name_value_len;
	char source_name_value[29];
	int az_start_len;
	char az_start[8];
	double az_start_value;
	int za_start_len;
	char za_start[8];
	double za_start_value;
	int src_raj_len;
	char src_raj[7];
	double src_raj_value;
	int src_dej_len;
	char src_dej[7];
	double src_dej_value;
	int tstart_len;
	char tstart[6];
	double tstart_value;
	int tsamp_len;
	char tsamp[5];
	double tsamp_value;
	int fch1_len;
	char fch1[4];
	double fch1_value;
	int foff_len;
	char foff[4];
	double foff_value;
	int nchans_len;
	char nchans[6];
	int nchans_value;
	int telescope_id_len;
	char telescope_id[12];
	int telescope_id_value;
	int machine_id_len;
	char machine_id[10];
	int machine_id_value;
	int data_type_len;
	char data_type[9];
	int data_type_value;
	int ibeam_len;
	char ibeam[5];
	int ibeam_value;
	int nbeams_len;
	char nbeams[6];
	int nbeams_value;
	int nbits_len;
	char nbits[5];
	int nbits_value;
	int barycentric_len;
	char barycentric[11];
	int barycentric_value;
	int pulsarcentric_len;
	char pulsarcentric[13];
	int pulsarcentric_value;
	int nifs_len;
	char nifs[4];
	int nifs_value;
	int HEADER_END_len;
	char HEADER_END[10];
};

double getMJD(struct full_time ftime);

void init_filerbank_header_buffer(struct filerbank_header& header, struct filerbank_header_buffer &buffer);

struct filerbank_header_buffer get_default_beam_filterbank_header();