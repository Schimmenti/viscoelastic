#include <list>
#include <set>
#include <iostream>
#include <cmath>
#include <random>
#include <limits>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip> 
#include <vector>
#include <list>
#include <stdio.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <map>

using namespace std;

default_random_engine av_engine(1204565);
normal_distribution<float> av_normal;
exponential_distribution<float> av_drop;

//float* saved_fth;
//float* saved_f;
//float* saved_g;
//float* saved_dr_times;
//float* saved_as_times;
//stringstream saved_av;
//stringstream saved_normal;
//stringstream saved_drop;



float pinf = std::numeric_limits<float>::infinity();
int mod(int a, int b)
{
    return (a % b + b) % b;
}

float get_threshold(float f1, float f2)
{
    return abs(f1+f2*av_normal(av_engine));
}
int get_neighbour_obc(int site, int nIndex, int Lx, int Ly)
{
    int x = site / Ly;
    int y = site % Ly;
    if(nIndex==0)
    {
        if(x==Lx-1)
            return -1;
        else
            return (x+1)*Ly+y;
    }
    else if(nIndex==1)
        if(x==0)
            return -1;
        else
            return (x-1)*Ly+y;
    else if(nIndex==2)
        if(y==Ly-1)
            return -1;
        else
            return x*Ly+y+1;
    else
        if(y==0)
            return -1;
        else
            return x*Ly+y-1;
};
int get_neighbour_rbc(int site, int nIndex, int Lx, int Ly)
{
    int x = site / Ly;
    int y = site % Ly;
    if(nIndex==0)
    {
        if(x==Lx-1)
            return site;
        else
            return (x+1)*Ly+y;
    }
    else if(nIndex==1)
        if(x==0)
            return site;
        else
            return (x-1)*Ly+y;
    else if(nIndex==2)
        if(y==Ly-1)
            return site;
        else
            return x*Ly+y+1;
    else
        if(y==0)
            return site;
        else
            return x*Ly+y-1;
};
int get_neighbour_pbc(int site, int nIndex, int Lx, int Ly)
{
    int x = site / Ly;
    int y = site % Ly;
    if(nIndex==0)
        return mod(x+1,Lx)*Ly+y;
    else if(nIndex==1)
        return mod(x-1,Lx)*Ly+y;
    else if(nIndex==2)
        return x*Ly+mod(y+1,Ly);
    else
        return x*Ly+mod(y-1,Ly);
};


int find_epicenter(float* fth, float* f, float* g, float* dr_times, float* as_times, int N, bool* is_as, float* dt)
{
    int as_epi = -1;
    int dr_epi = -1;
    float as_max = 0;
    float dr_min = pinf;

    for(int i = 0; i < N; i++)
    {
        if(as_times[i] > 0 && as_times[i] < 1 && as_times[i] > as_max)
        {
            as_epi = i;
            as_max = as_times[i];
        }
        if(dr_times[i]< dr_min)
        {
            dr_epi = i;
            dr_min = dr_times[i];
        }
    }

    if(as_epi != -1)
    {
        *is_as = true;
        *dt = as_max;
        return as_epi;
    }
    else
    {
        *is_as = false;
        *dt = dr_min;
        return dr_epi;
    }
}

void drive(float* fth, float* f, float* g, float* dr_times, float* as_times, int N,int dr_epi, float dr_min)
{
    for(int i = 0; i < N; i++)
        {
            f[i] = 0;
            g[i]  += dr_min;
            dr_times[i] = fth[i]-g[i];
            as_times[i] = pinf;
            //if(save_state)
            //{
            //    saved_fth[i] = fth[i];
            //    saved_f[i] = f[i];
            //    saved_g[i] = g[i];
            //    saved_dr_times[i] = dr_times[i];
            //    saved_as_times[i] = as_times[i];
            //}
        }
    //if(save_state)
    //{
    //    saved_av.str(std::string());
    //    saved_normal.str(std::string());
    //    saved_drop.str(std::string());
    //    saved_av << av_engine;
    //    saved_normal << av_normal;
    //    saved_drop << av_drop;
    //}
    g[dr_epi] = fth[dr_epi];
}
void aftershock(float* fth, float* f, float* g, float* dr_times, float* as_times, int N,int as_epi, float as_max)
{
for(int i = 0; i < N; i++)
        {
            f[i] *= as_max;
            dr_times[i] = fth[i]-g[i];
            if(f[i]<0)
            {
                float new_as = (fth[i]-g[i])/f[i];
                if(new_as>0 && new_as < 1)
                {
                    as_times[i] = (fth[i]-g[i])/f[i];
                }
                else
                {
                    as_times[i] = pinf;
                }
            }
            else
            {
                as_times[i] = pinf;
            }
        }
        f[as_epi] = fth[as_epi]-g[as_epi];
}



int nanargmin(float* x, int N, float min_val=0.0)
{
    float temp =min_val;
    int idx = -1;
    for(int i = 0; i < N; i++)
    {
        if(x[i] != pinf && x[i] >= min_val && x[i] < temp)
        {
            idx = i;
            temp = x[i];
        }
    }
    return idx;
}
set<int>* propagate(float k0,float k1, float k2, float f1, float f2, int Lx, int Ly, float dh,
    float* fth, float* f, float* g, int epicenter, int* S, float* S_real, int* A, set<int>* sequence_sites, map<int, float>& s_sequence)
    {
        set<int>* sites;
        sites = new set<int>();
        sites->insert(epicenter);
        set<int>* touchedSites;
        touchedSites = new set<int>();
        set<int>* avalancheSites;
        avalancheSites = new set<int>();
        set<int>* new_sites;
        new_sites = new set<int>();
        *S=0;
        *S_real=0.0;

        while(sites->size()>0)
        {
            float z = dh*av_drop(av_engine);
            //float z = dh;
            for(int site : *sites)
            {
                sequence_sites->insert(site);
                float old_drop = s_sequence[site];
                s_sequence[site] = old_drop + z;
                *S = *S+1;
                *S_real += z;
                fth[site] = get_threshold(f1,f2);
                f[site] -= 4*k2*z;
                g[site] -= (4*k1+k0)*z;
                touchedSites->insert(site);
                avalancheSites->insert(site);
                if(fth[site]-f[site]-g[site] <= 0)
                    new_sites->insert(site);
            }
            for(int site : *sites)
            {
                //int x = site / Ly;
                //int y = site % Ly;
                for(int j =0 ; j < 4; j++)
                {
                    //int n_site = get_neighbour_obc(site,j,Lx,Ly);
                    int n_site = get_neighbour_rbc(site,j,Lx,Ly);
                    if(n_site != -1)
                    {
                        //int xj = n_site / Ly;
                        //int yj = n_site % Ly;
                        touchedSites->insert(n_site);
                        
                        f[n_site] += k2*z;
                        g[n_site] += k1*z;
                        if(fth[n_site]-f[n_site]-g[n_site] <= 0)
                            new_sites->insert(n_site);
                    }
                }
            }
            delete sites;
            sites = new_sites;
            new_sites = new set<int>();
        }

        //int invalid_cnt = 0;
        //    for(int i = 0; i < Lx*Ly; i++)
        //    {
        //        if(fth[i]-f[i]-g[i] <= 0)
        //        {
        //            cout << fth[i]-f[i] - g[i] << endl;
        //            invalid_cnt++;
        //        }
        //    }
        //    cout << invalid_cnt << " " << new_sites->size() << endl;
//
        *A = avalancheSites->size();
        delete sites;
        delete new_sites;
        return touchedSites;
    };
void snapshot(string hash_fname_str, int snap_idx, float* fth, float* f, float* g, int N)
{
    string folder = "vdep_" + hash_fname_str;
    ofstream file1( folder + "/snap_" + to_string(snap_idx) + "_gen_seed.dat");
    ofstream file2( folder + "/snap_" + to_string(snap_idx) +"_fth_seed.dat");
    ofstream file3( folder + "/snap_" + to_string(snap_idx) +"_drop_seed.dat");
    file1 << av_engine;
    file2 << av_normal;
    file3 << av_drop;
    file1.close();
    file2.close();
    file3.close();
    ofstream snaps(folder + "/snap_" + to_string(snap_idx) + ".dat", ios::binary | ios::out   );
    snaps.write((char*)fth, sizeof(float)*N);
    snaps.write((char*)f, sizeof(float)*N);
    snaps.write((char*)g, sizeof(float)*N);
    //snaps.write((char*)as_times, sizeof(float)*N);
    //snaps.write((char*)dr_times, sizeof(float)*N);
    snaps.close();
}

void get_snapshot(string hash_fname_str, int snap_idx, float* fth, float* f, float* g, int N)
{
    string folder = "vdep_" + hash_fname_str;
    ifstream file1( folder + "/snap_" + to_string(snap_idx) + "_gen_seed.dat");
    ifstream file2( folder + "/snap_" + to_string(snap_idx) +"_fth_seed.dat");
    ifstream file3( folder + "/snap_" + to_string(snap_idx) +"_drop_seed.dat");
    file1 >> av_engine;
    file2 >> av_normal;
    file3 >> av_drop;
    file1.close();
    file2.close();
    file3.close();
    ifstream snaps(folder + "/snap_" + to_string(snap_idx) + ".dat", ios::binary | ios::in   );
    snaps.read((char*)fth, sizeof(float)*N);
    snaps.read((char*)f, sizeof(float)*N);
    snaps.read((char*)g, sizeof(float)*N);
    //snaps.write((char*)as_times, sizeof(float)*N);
    //snaps.write((char*)dr_times, sizeof(float)*N);
    snaps.close();
}



unordered_map<string,string> process_arguments(int argc, char** argv)
{
    std::string delimiter = "=";
    unordered_map<string,string> dict;
    for(int i = 1; i < argc; i++)
    {
        string str = string(argv[i]);
        size_t pos = str.find(delimiter);
        string name = str.substr(0,pos);
        string value = str.substr(pos+1);
        dict.insert(make_pair(name,value));
    }
    return dict;
}


void iso_stress_distribution(string iso_fname, string iso_out, int iso_cnt, float k0,float k1, float k2, float f1, float f2, int Lx, int Ly, float dh)
{
    
    int N = Lx*Ly;
    float* dr_times = new float[N];
    float* as_times = new float[N];
    float* fth = new float[N];
    float* f = new float[N];
    float* g = new float[N];
    ifstream snaps(iso_fname, ios::binary | ios::in   );
    if(!snaps.is_open())
    {
        cout << "FATAL" << endl;
        exit(0);
    }
    snaps.read((char*)fth, sizeof(float)*N);
    snaps.read((char*)f, sizeof(float)*N);
    snaps.read((char*)g, sizeof(float)*N);
    snaps.close();
    bool is_as = false;
    float waiting_time = 0.0;
    float* fraction = new float[N];
    float* s_fraction = new float[N];
    for(int i = 0; i < N; i++)
    {
        dr_times[i]= fth[i]-g[i];
        if(f[i]<0)
        {
            float new_as = (fth[i]-g[i])/f[i];
            if(new_as>0)
            {
                as_times[i] = new_as;
            }
            else
            {
                as_times[i] = pinf;
            }
        }
        else
        {
            as_times[i] = pinf;
        }
        fraction[i] = 0;
        s_fraction[i] = 0;
    }
    int epicenter = find_epicenter(fth,f,g,dr_times,as_times,N, &is_as, &waiting_time);
    if(!is_as){
        drive(fth,f,g,dr_times,as_times,N,epicenter,waiting_time);
    }
    else
    {
        aftershock(fth,f,g,dr_times,as_times,N,epicenter,waiting_time);
    }


    for(int c = 0; c < iso_cnt; c++)
    {
        set<int> sequence_sites;
        map<int,float> s_sequence_sites;
        int S = 0;
        float S_real = 0.0;
        int A = 0;
        float* fth_here = new float[N];
        float* f_here = new float[N];
        float* g_here = new float[N];
        for(int i = 0; i < N; i++)
        {
            fth_here[i] = fth[i];
            f_here[i] = f[i];
            g_here[i] = g[i];
        }
        set<int>* touchedSites = propagate(k0,k1,k2,f1,f2,Lx,Ly,dh,fth_here,f_here,g_here,epicenter, &S, &S_real, &A, &sequence_sites,s_sequence_sites);
        for(int j : sequence_sites)
        {
            fraction[j] += 1.0/iso_cnt;
        }
        for(pair<int, float> p : s_sequence_sites)
        {
            s_fraction[p.first]  = p.second/iso_cnt;
        }
        delete fth_here, f_here, g_here;
    }

    ofstream snaps0(iso_out, ios::binary | ios::out   );
    snaps0.write((char*)fraction, sizeof(float)*N);
    snaps0.write((char*)s_fraction, sizeof(float)*N);
    snaps0.close();

    delete dr_times, as_times, fth,f,g;
    delete fraction, s_fraction;
}

void iso_stress_sequence(string iso_fname, string iso_out, int iso_cnt, float k0,float k1, float k2, float f1, float f2, int Lx, int Ly, float dh)
{
    
    int N = Lx*Ly;
    float* dr_times = new float[N];
    float* as_times = new float[N];
    float* fth = new float[N];
    float* f = new float[N];
    float* g = new float[N];
    ifstream snaps(iso_fname, ios::binary | ios::in   );
    snaps.read((char*)fth, sizeof(float)*N);
    snaps.read((char*)f, sizeof(float)*N);
    snaps.read((char*)g, sizeof(float)*N);
    snaps.close();
    bool is_as = false;
    float waiting_time = 0.0;
    float* fraction = new float[N];
    float* s_fraction = new float[N];
    for(int i = 0; i < N; i++)
    {
        dr_times[i]= fth[i]-g[i];
        if(f[i]<0)
        {
            float new_as = (fth[i]-g[i])/f[i];
            if(new_as>0)
            {
                as_times[i] = new_as;
            }
            else
            {
                as_times[i] = pinf;
            }
        }
        else
        {
            as_times[i] = pinf;
        }
        fraction[i] = 0;
        s_fraction[i] = 0;
    }
    int epicenter = find_epicenter(fth,f,g,dr_times,as_times,N, &is_as, &waiting_time);
    if(!is_as){
        drive(fth,f,g,dr_times,as_times,N,epicenter,waiting_time);
    }
    else
    {
        aftershock(fth,f,g,dr_times,as_times,N,epicenter,waiting_time);
    }


    for(int c = 0; c < iso_cnt; c++)
    {
        set<int> sequence_sites;
        map<int,float> s_sequence_sites;
        int S = 0;
        float S_real = 0.0;
        int A = 0;
        float* fth_here = new float[N];
        float* f_here = new float[N];
        float* g_here = new float[N];
        for(int i = 0; i < N; i++)
        {
            fth_here[i] = fth[i];
            f_here[i] = f[i];
            g_here[i] = g[i];
        }
        set<int>* touchedSites = propagate(k0,k1,k2,f1,f2,Lx,Ly,dh,fth_here,f_here,g_here,epicenter, &S, &S_real, &A, &sequence_sites,s_sequence_sites);
        for(int j : sequence_sites)
        {
            fraction[j] += 1.0/iso_cnt;
        }
        for(pair<int, float> p : s_sequence_sites)
        {
            s_fraction[p.first]  = p.second/iso_cnt;
        }
    }

    ofstream snaps0(iso_out, ios::binary | ios::out   );
    snaps0.write((char*)fraction, sizeof(float)*N);
    snaps0.write((char*)s_fraction, sizeof(float)*N);
    snaps0.close();
}


int main(int argc, char **argv)
{

    int Lx = 200;
    int Ly = 200;
    
    float k0 = 0.01;
    float k1 = 0.0;
    float k2 = 1.0;
    float dh = 0.1;
    float f1 = 1;
    float f2 = 1;


    int snapshot_flag = 0;
    int snapshot_period = 0;
    int load_snapshot = -1;

    string source_file = "";
    string iso_source_folder = "";
    string iso_output_folder = "";
    int iso_stress_count = 0;
    int iso_resume_from  = -1;
    
    int n_events = 10000000;

    //arguments processing

    list<int> events_to_save;
    bool save_events = false;

    if(argc==1)
    {
        cout << "Arguments must be specified in the form arg_name=arg_value" << endl;
        cout << "List of arguments: " << endl;
        cout << "Lx : rows" << endl;
        cout << "Ly : columns" << endl;
        cout << "k0 : dissipation" << endl;
        cout << "k1 : elasticity" << endl;
        cout << "k2 : relaxation elasticity" << endl;
        cout << "dh : average slip" << endl;
        cout << "f1 : average threshold" << endl;
        cout << "f2 : std threshold" << endl;
        cout << "period : snapshot period. Leave or put to 0 to deactivate." << endl;
        cout << "load_snapshot : snapshot index to load" << endl;
        cout << "n_events : number of events to generate." << endl;
        exit(0);
    }

    unordered_map<string,string> args = process_arguments(argc,argv);
    for (std::pair<string, string> element : args)
    {
        string name = element.first;
        if(name == "k0")
        {
            k0 = stof(element.second);
        }
        else if (name=="k1")
        {
            k1 = stof(element.second);
        }
        else if (name=="k2")
        {
            k2 = stof(element.second);
        }
        else if (name=="dh")
        {
            dh = stof(element.second);
        }
        else if (name=="f1")
        {
            f1 = stof(element.second);
        }
        else if (name=="f2")
        {
            f2 = stof(element.second);
        }
        else if (name=="Lx")
        {
            Lx = stoi(element.second);
        }
        else if (name=="Ly")
        {
            Ly = stoi(element.second);
        }
        else if (name=="period" | name == "snapshot_period")
        {
            snapshot_flag = 1;
            snapshot_period = stoi(element.second);
        }
        else if(name=="load_snapshot")
        {
            load_snapshot =  stoi(element.second);
        }
        else if(name=="n_events")
        {
            n_events =  stoi(element.second);
        }
        else if(name=="source_file")
        {
            source_file = element.second;
        }
        else if(name=="iso_source_folder")
        {
            iso_source_folder = element.second;
        }
        else if(name=="iso_output_folder")
        {
            iso_output_folder = element.second;
        }
        else if(name=="iso_stress_count")
        {
            iso_stress_count =  stoi(element.second);
        }
        else if(name=="iso_resume_from")
        {
            iso_resume_from = stoi(element.second);
        }
    }

    if(iso_stress_count  > 0)
    {
        cout << iso_source_folder << endl;
        cout << iso_output_folder << endl;
        cout << source_file << endl;
        cout << "Creating distribution (isostress)..." << endl;
        std::ifstream infile(source_file);
        int file_id;
        list<int> events_to_run;
        while(infile >> file_id)
        {
            events_to_run.push_back(file_id);
        }
        infile.close();
        for (int file_id : events_to_run)
        {
            if(file_id < iso_resume_from)
                continue;
            cout << file_id << endl;
            string iso_source_file = iso_source_folder + "events0_" + to_string(file_id) + ".dat";
            string iso_output_file = iso_output_folder + "distr0_" + to_string(file_id) + ".dat";
            iso_stress_distribution(iso_source_file, iso_output_file, iso_stress_count, k0,k1,k2,f1,f2,Lx,Ly,dh);
        }
        exit(0);
    }

    int max_events_to_save_per_file = 1000;
    int events_file_idx = 0;
    if(source_file != "")
    {
        std::ifstream lines(source_file);
        int a;
        while(lines >> a)
        {
            events_to_save.push_back(a);
        }
        //events_to_save.resize(max_events_to_save);
        cout << "Files to save: " << events_to_save.size() << endl;
        save_events = true;
    }

    //prepare system
    hash<string> hasher;

    //unique system config.
    string hash_name = "";
    hash_name +=  to_string(Lx);
    hash_name +=  to_string(Ly);
    hash_name +=  to_string(k0);
    hash_name +=  to_string(k1);
    hash_name +=  to_string(k2);
    hash_name +=  to_string(dh);
    hash_name +=  to_string(f1);
    hash_name +=  to_string(f2);

    size_t hash_fname = hasher(hash_name);

    string hash_fname_str = to_string(hash_fname);


    int N = Lx*Ly;
    
    int seq_length = 0;
    float* fth = new float[N];
    float* f = new float[N];
    float* g = new float[N];
    float* dr_times = new float[N];
    float* as_times = new float[N];
    set<int> sequence_sites; //to compute A_seq
    map<int,float> s_sequence_sites; //to compute S_seq

    //saved_fth = new float[N];
    //saved_f = new float[N];
    //saved_g = new float[N];
    //saved_dr_times = new float[N];
    //saved_as_times = new float[N];
    

    if(load_snapshot>-1)
    {
        cout << "Restarting simulation..." << endl;
        get_snapshot(hash_fname_str, load_snapshot, fth, f, g, N);
        for(int i = 0; i < N; i++)
        {
            dr_times[i]= fth[i]-g[i];
            if(f[i]<0)
            {
                float new_as = (fth[i]-g[i])/f[i];
                if(new_as>0)
                {
                    as_times[i] = new_as;
                }
                else
                {
                    as_times[i] = pinf;
                }
            }
            else
            {
                as_times[i] = pinf;
            }
        }
    }
    else
    {
        cout << "Starting simulation from scratch..." << endl;
        for(int i = 0; i < N; i++)
        {
            fth[i] = get_threshold(f1,f2);
            f[i] = 0;
            g[i] = 0;
            dr_times[i]= fth[i];
            as_times[i] = pinf;
        }
    }

    
    //save initial config

    mkdir(("vdep_" + hash_fname_str).c_str(), 0777);
    ofstream params;
    params.open( "vdep_" +  hash_fname_str + "/params.txt", ios::out); //self consistent with the hash!
    params << "Lx=" << Lx << endl;
    params << "Ly=" << Ly << endl;
    params << "k0=" << setprecision(5) << k0 << endl;
    params << "k1=" << setprecision(5) << k1 << endl;
    params << "k2=" << setprecision(5) << k2 << endl;
    params << "dh=" << setprecision(5) << dh << endl;
    params << "f1=" << setprecision(5) << f1 << endl;
    params << "f2=" << setprecision(5) << f2 << endl;
    
    params.close();
    ofstream outfile;
    outfile.open( "vdep_" + hash_fname_str + "/data.txt", ios::out);
    outfile << "Sr S A AS DT EC" << endl;

    ofstream seq_file;
    seq_file.open( "vdep_" + hash_fname_str + "/a_seq.txt", ios::out);
    seq_file << "L Aseq" << endl;

    int snap_idx = 0;
    
    //if(snapshot_flag>0 &&  snapshot_period > 0)
    //    snapshot(hash_fname_str,snap_idx,fth,f,g,N);

    string folder = "vdep_" + hash_fname_str;
    //ofstream snaps0(folder + "/events0_" + to_string(events_file_idx) + ".dat", ios::binary | ios::out   );
    //ofstream snaps1(folder + "/events1_" + to_string(events_file_idx) + ".dat", ios::binary | ios::out   );
    //ofstream snaps_ndx(folder + "/events_index_" + to_string(events_file_idx) + ".txt", ios::out   );

    


    for(int ndx = 0; ndx < n_events; ndx++)
    {
        if(save_events && events_to_save.front() == ndx)
        {
            cout << "Saving (0): " << ndx << endl;
            ofstream snaps0(folder + "/events0_" + to_string(ndx) + ".dat", ios::binary | ios::out   );
            snaps0.write((char*)fth, sizeof(float)*N);
            snaps0.write((char*)f, sizeof(float)*N);
            snaps0.write((char*)g, sizeof(float)*N);
            snaps0.close();
            //snaps_ndx << ndx << endl;
        }


        bool is_as = false;
        float waiting_time = 0.0;
        int epicenter = find_epicenter(fth,f,g,dr_times,as_times,N, &is_as, &waiting_time);
        if(!is_as){
            seq_file << seq_length << " " << sequence_sites.size() << endl;
            sequence_sites.clear(); //clear for new sequence
            s_sequence_sites.clear();
            seq_length = 1;
            drive(fth,f,g,dr_times,as_times,N,epicenter,waiting_time);
            }
            else
            {
                aftershock(fth,f,g,dr_times,as_times,N,epicenter,waiting_time);
                seq_length++;
            }

        
        

        int S = 0;
        float S_real = 0.0;
        int A = 0;
        set<int>* touchedSites = propagate(k0,k1,k2,f1,f2,Lx,Ly,dh,fth,f,g,epicenter, &S, &S_real, &A, &sequence_sites,s_sequence_sites);
        outfile << std::setprecision(15) << S_real << " ";
        outfile << S << " " << A << " ";
        outfile << is_as << " ";
        outfile << std::setprecision(15) << waiting_time << " ";
        outfile << epicenter << endl;
        for(int i : *touchedSites)
        {
            dr_times[i] = fth[i]-g[i];
            
            if(f[i]<0)
            {
                float new_as = (fth[i]-g[i])/f[i];
                if(new_as>0)
                {
                    as_times[i] = new_as;
                }
                else
                {
                    as_times[i] = pinf;
                }
            }
            else
            {
                as_times[i] = pinf;
            }
        }
        delete touchedSites;

        //everything is up to date, one can safely save the state
        if(snapshot_flag > 0 && snapshot_period > 0 &&  ndx % snapshot_period == 0)
        {
            cout << "Saving at: " << ndx << endl;
            snap_idx++;
            snapshot(hash_fname_str,snap_idx,fth,f,g,N);
        }


        if(save_events && events_to_save.front() == ndx)
        {
            cout << "Saving (1): " << ndx << endl;
            events_to_save.pop_front();
            ofstream snaps1(folder + "/events1_" + to_string(ndx) + ".dat", ios::binary | ios::out   );
            snaps1.write((char*)fth, sizeof(float)*N);
            snaps1.write((char*)f, sizeof(float)*N);
            snaps1.write((char*)g, sizeof(float)*N);
            snaps1.close();
            if(events_to_save.size() == 0)
            {
                cout << "Finished saving files. Exiting...." << endl;
                outfile.close();
                seq_file.close();
                //snaps0.close();
                //snaps1.close();
                //snaps_ndx.close();
                exit(0);
            }
            //if(events_to_save.size() % max_events_to_save_per_file == 0)
            //{
            //    cout << "Switching file...." << endl;
            //    snaps0.close();
            //    snaps1.close();
            //    snaps_ndx.close();
            //    events_file_idx++;
            //    snaps0.open(folder + "/events0_" + to_string(events_file_idx) + ".dat", ios::binary | ios::out   );
            //    snaps1.open(folder + "/events1_" + to_string(events_file_idx) + ".dat", ios::binary | ios::out   );
            //    snaps_ndx.open(folder + "/events_index_" + to_string(events_file_idx) + ".txt", ios::out   );
            //} 
            
        }


        if(ndx % 10000 == 0)
        {
            outfile.close();
            outfile.open( "vdep_" + hash_fname_str + "/data.txt", ios::out | ios::app);
        }

        
    }

    outfile.close();
    seq_file.close();
    return 0;
};
