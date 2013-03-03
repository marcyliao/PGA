// a fast genetic algorithm for the 0-1 knapsack problem
// by karoly zsolnai - keeroy@cs.bme.hu
// test case: 1000 items, 50 knapsack size
//
// compilation by: g++ genetic.cpp -O3 -ffast-math -fopenmp
#include <math.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <limits.h>
#include <cstdlib>

#include <iostream>

using namespace std;

#if defined(__linux) || defined(__linux__)
	unsigned int seed = time(NULL);
	#define RND ((double)rand_r(&seed)/RAND_MAX) // reentrant uniform rnd
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
	#define RND ((double)rand()/RAND_MAX) // uniform rnd
#endif

using namespace std;

//lock
pthread_mutex_t lock;

struct chromo {
	chromo(int dimc) { items = new bool[dimc]; }
	~chromo() {	items = NULL; }
	void mutate(const int dimc, const int count) {
		int mi;
		for(int i=0;i<count;i++) {
			mi = (int)(round(RND*(dimc-1)));
			items[mi] = !items[mi];
		}
	}
	bool* items;
	int f;
};

struct Param 
{
    int number_of_thread;
    int index;
    double crp;
    int ind;
    int ind2;
    int parc;
    
    double avg;
    
    int limit; // knapsack weight limit
	int pop; // chromosome population size
	int gens; // maximum number of generations
	int disc; // chromosomes discarded via elitism
    int dimw;
    
    vector<chromo> ch;
    vector<int> w,v;
    
    int best;
    
};


int fitness(bool*& x, const int dimc, const vector<int>& v, const vector<int>& w, const int limit) {
	int fit = 0, wsum = 0;
	for(int i=0;i<dimc;i++) {
		wsum += x[i]*w[i];
		fit += x[i]*v[i];
	}
    //cout<<"wsum "<<wsum<<endl;
	if(wsum>limit) fit -= 7*(wsum-limit); // penalty for invalid solutions
	return fit;
}

void crossover1p(const chromo& c1, const chromo& c2, const chromo& c3, const int dimc, const int cp) {
	for(int i=0;i<dimc;i++) {
		if(i<cp) { c3.items[i] = c1.items[i]; }
		else { c3.items[i] = c2.items[i]; }
	}
}

void crossover1p_b(const chromo &c1, const chromo &c2, const chromo &c3, int dimc, int cp) {
	for(int i=0;i<dimc;i++) {
		if(i>=cp) { c3.items[i] = c1.items[i]; }
		else { c3.items[i] = c2.items[i]; }
	}
}

void crossoverrand(const chromo &c1, const chromo &c2, const chromo &c3, const int dimc) {
	for(int i=0;i<dimc;i++) {
		if(round(RND)) { c3.items[i] = c1.items[i]; }
		else { c3.items[i] = c2.items[i]; }
	}
}

void crossoverarit(const chromo &c1, const chromo &c2, const chromo &c3, int dimc) {
	for(int i=0;i<dimc;i++) {
		c3.items[i] = (c1.items[i]^c2.items[i]);
	}
}

bool cfit(const chromo &c1,const chromo &c2) { return c1.f > c2.f; }
bool cmpfun(const std::pair<int,double> &r1, const std::pair<int,double> &r2) { return r1.second > r2.second; }

int coin(const double crp) { // a cointoss
	if(RND<crp) return 1; // crossover
	else return 0; // mutation
}

// initializes the chromosomes with the results of a greedy algorithm
void initpopg(bool**& c, const std::vector<int> &w, const std::vector<int> &v, const int dimw, const int limit, const int pop) {
	std::vector<std::pair<int,double> > rvals(dimw);
	std::vector<int> index(dimw,0);
	for(int i=0;i<dimw;i++) {
		rvals.push_back(std::pair<int,double>(std::make_pair(i,(double)v[i]/(double)w[i])));
	}
	std::sort(rvals.begin(),rvals.end(),cmpfun);
	int currentw = 0, k;
	for(int i=0;i<dimw;i++) {
		k = rvals[i].first;
		if(currentw + w[k] <= limit) { // greedy fill
			currentw += w[k];
			index[k] = 1;
		}
	}
	for(int i=0;i<pop;i++) {
		for(int j=0;j<dimw;j++) {
			c[i][j] = index[j];
		}
	}
}

void threadFunction(Param* param)
{
    
    //Param *param = (Param*) data;
    
    
    //release data spawns
    
    
    int size = param->pop/param->number_of_thread;
    int index = param->index;
    int begin = index*size;
    int end = begin+size-1;
    int local_best = 0;
    //cout<<endl<<"range: "<<" "<<size<<" "<<begin<<" "<<end<<" "<<param->best<<endl;
    int dimw = (int)param->dimw;
    for(int i=begin;i<end;i++)
    {
        if(i>param->pop-param->disc) { // elitism - only processes the discarded chromosomes
            if(coin(param->crp)==1) { // crossover section
                param->ind = param->parc+round(10*RND); // choosing parents for crossover
                param->ind2 = param->parc+1+round(10*RND);
                // choose a crossover strategy here
                crossover1p(param->ch[param->ind%param->pop],param->ch[param->ind2%param->pop],param->ch[i],dimw,round(RND*(dimw-1)));
                //					crossoverrand(ch[ind],ch[ind2],ch[i],dimw);
                //					crossoverarit(ch[0],ch[1],ch[i],dimw);
                param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, param->limit);
                param->parc += 1;
            }
            else { // mutation section
                param->ch[i].mutate(dimw,1);
                param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, param->limit);
            }
        }
        param->avg += param->ch[i].f;
        /*
        if(param->ch[i].f>param->best)
            param->best=param->ch[i].f;
         */
        if (param->ch[i].f>local_best)
            local_best = param->ch[i].f;
        
        pthread_mutex_lock(&lock);
            if(param->best<local_best)
                param->best = local_best;
        pthread_mutex_unlock(&lock);
    }
    
    
    
    
    
    /*
    int size = param->pop/param->number_of_thread;
    
    int begin = index*size;
    int end = begin+size-1;
    cout<<endl<<"range: "<<index<<" "<<size<<" "<<begin<<" "<<end<<" "<<param->best<<endl;
    int dimw = (int)param->dimw;
    
     
     
     
     
    for(int i=begin;i<end;i++) {
        if(i>param->pop-param->disc)
        { // elitism - only processes the discarded chromosomes
            if(coin(param->crp)==1) { // crossover section
                param->ind = param->parc+round(10*RND); // choosing parents for crossover
                param->ind2 = param->parc+1+round(10*RND);
                // choose a crossover strategy here
                crossover1p(param->ch[param->ind%param->pop],param->ch[param->ind2%param->pop],param->ch[i],dimw,round(RND*(dimw-1
                                                                           )));
                //                              crossoverrand(ch[ind],ch[ind2],ch[i],dimw);
                //                              crossoverarit(ch[0],ch[1],ch[i],dimw);
                param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, param->limit);
                param->parc += 1;
                if (param->ch[i].f<0) cout<<"crossover "<<param->ch[i].f<<endl;
            }
            else { // mutation section
                param->ch[i].mutate(dimw,1);
                param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, param->limit);
                if (param->ch[i].f<0) cout<<"mutate "<<param->ch[i].f<<endl;
            }
        }
        param->avg += param->ch[i].f;
        //cout<<"compare: "<<param.best<<" "<<param.ch[i].f<<endl;
        if(param->ch[i].f>param->best)
        {
            //cout<<i<<" "<<"before best "<<param->best<<endl;
            param->best=param->ch[i].f;
            //cout<<i<<" "<<"after best "<<param->best<<endl;
        }
    }
    
//            for(int i=0;i<param.pop;i++) {
//                cout<<param.ch[i].f<<endl;
//            }

    */
}




int main(int argc, const char* argv[])
{
    
    if (argc<2)
    {
        cout<<"Invalid Arguments"<<endl;
        exit(-1);
    }
    
    Param *param = new Param;
    printf("Start \n");
    //cout<<atoi(argv[0]);
    param->number_of_thread = atoi(argv[1]);
    
    pthread_t *thread = new pthread_t[param->number_of_thread];
    
    pthread_mutex_init(&lock, NULL);
    
	
	srand(time(NULL));
//	vector<int> w, v; // items weights and values
	int info=0;
	FILE *f = fopen("10000_weights.txt","r");
	FILE *f2 = fopen("10000_values.txt","r");
	while(!feof(f) || !feof(f2) ) {
		fscanf(f," %d ",&info);
		param->w.push_back(info);
		info=0;
		fscanf(f2," %d ",&info);
		param->v.push_back(info);
	} // omitted fclose(f1) and fclose(f2) on purpose
    
	const int limit = 50; // knapsack weight limit
	const int pop = 256; // chromosome population size
	const int gens = INT_MAX; // maximum number of generations
	const int disc = (int)(ceil(pop*0.8)); // chromosomes discarded via elitism
	const int dimw = (int)param->w.size();
    param->limit = limit;
    param->pop = pop;
    param->gens = gens;
    param->disc = disc;
    param->dimw = dimw;
    
    
	int best = 0, ind = 0, ind2 = 0; // a few helpers for the main()
	int parc = 0; // parent index for crossover
    param->best = best;
    param->ind = ind;
    param->ind2 = ind2;
    param->parc = parc;
    
	double avg = 0, crp = 0.35; // crossover probability
    param->avg = avg;
    param->crp = crp;
    
	//vector<chromo> ch(pop,chromo(dimw));
    param->ch = vector<chromo>(pop,chromo(dimw));
    
	bool **c = new bool*[pop];
	for(int i=0;i<pop;i++) c[i] = new bool[dimw];
    
	clock_t start = clock();
    long st = time(0);
	printf("Initializing population with a greedy algorithm...");
	initpopg(c,param->w,param->v,dimw,limit,pop);
    
//    for (int i = 0; i<param.dimw; i++)
//        cout<<param.w[i]<<endl;
    
	printf("done!");
    
	for(int i=0;i<pop;i++) {
		param->ch[i].items = c[i];
		param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, limit);
        //cout<<param->ch[i].f<<endl;
	}
    param->best = 10;
    //cout<<"best "<<param->best<<endl;
	printf("\n\n");

	for(int p=0;p<gens;p++) {
		std::sort(param->ch.begin(), param->ch.end(), cfit);
        
        for (int i = 0; i<param->number_of_thread; i++)
        {
            param->index = i;
            pthread_create(&thread[i], NULL, (void *(*)(void *))threadFunction, (void*)param);
        }
        
        for (int i = 0; i<param->number_of_thread; i++)
            pthread_join(thread[i], NULL);
//        for (int i = 0; i<param->number_of_thread; i++)
//        {
//            param->index = i;
//            threadFunction(param);
//        }
		
//        for(int i=0;i<pop;i++) {
//            cout<<param.ch[i].f<<endl;
//        }
        /*
        int size = param->pop/param->number_of_thread;
        
        int begin = 0;//index*size;
        int end = begin+size-1;
        cout<<endl<<"range: "<<" "<<size<<" "<<begin<<" "<<end<<" "<<param->best<<endl;
        //int dimw = (int)param->dimw;
        for(int i=0;i<pop;i++) {
            if(i>param->pop-param->disc) { // elitism - only processes the discarded chromosomes
                if(coin(param->crp)==1) { // crossover section
                    param->ind = param->parc+round(10*RND); // choosing parents for crossover
                    param->ind2 = param->parc+1+round(10*RND);
                    // choose a crossover strategy here
                    crossover1p(param->ch[param->ind%pop],param->ch[param->ind2%pop],param->ch[i],dimw,round(RND*(dimw-1)));
                    //					crossoverrand(ch[ind],ch[ind2],ch[i],dimw);
                    //					crossoverarit(ch[0],ch[1],ch[i],dimw);
                    param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, param->limit);
                    param->parc += 1;
                }
                else { // mutation section
                    param->ch[i].mutate(dimw,1);
                    param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, param->limit);
                }
            }
            param->avg += param->ch[i].f;
            if(param->ch[i].f>param->best)
                param->best=param->ch[i].f;
        }
        */
        
        //            for(int i=0;i<param.pop;i++) {
        //                cout<<param.ch[i].f<<endl;
        //            }
        
		param->parc = 0;
		if(p%5==0) {
//			printf("\n#%d\t",p);
//			printf("best fitness: %d \t",param->best);
//			printf("avg fitness: %f",param->avg/param->pop);
			//if(param->best == 675||p>3000) goto end; // psst...don't tell anyone
            if(p>500) goto end;
		}
		param->best = param->avg = 0;
	}

end:
	printf("\n\n");
	clock_t end = clock();
    long ter = time(0);
	double t = (double)(ter-st);//(end-start)/CLOCKS_PER_SEC;
	printf("\nCompletion time: %fs.\n",t);
	return 0;
}


/*
 for(int i=0;i<pop;i++) {
 if(i>pop-disc) { // elitism - only processes the discarded chromosomes
 if(coin(crp)==1) { // crossover section
 ind = parc+round(10*RND); // choosing parents for crossover
 ind2 = parc+1+round(10*RND);
 // choose a crossover strategy here
 crossover1p(ch[ind%pop],ch[ind2%pop],ch[i],dimw,round(RND*(dimw-1)));
 //					crossoverrand(ch[ind],ch[ind2],ch[i],dimw);
 //					crossoverarit(ch[0],ch[1],ch[i],dimw);
 ch[i].f = fitness(ch[i].items, dimw ,v, w, limit);
 parc += 1;
 }
 else { // mutation section
 ch[i].mutate(dimw,1);
 ch[i].f = fitness(ch[i].items, dimw ,v, w, limit);
 }
 }
 avg += ch[i].f;
 if(ch[i].f>best) best=ch[i].f;
 }
*/

/*
 for(int i=begin;i<end;i++) {
 if(i>param->pop-param->disc)
 { // elitism - only processes the discarded chromosomes
 if(coin(param->crp)==1) { // crossover section
 param->ind = param->parc+round(10*RND); // choosing parents for crossover
 param->ind2 = param->parc+1+round(10*RND);
 // choose a crossover strategy here
 crossover1p(param->ch[param->ind%param->pop],param->ch[param->ind2%param->pop],param->ch[i],dimw,round(RND*(dimw-1
 )));
 //                              crossoverrand(ch[ind],ch[ind2],ch[i],dimw);
 //                              crossoverarit(ch[0],ch[1],ch[i],dimw);
 param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, param->limit);
 param->parc += 1;
 if (param->ch[i].f<0) cout<<"crossover "<<param->ch[i].f<<endl;
 }
 else { // mutation section
 param->ch[i].mutate(dimw,1);
 param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, param->limit);
 if (param->ch[i].f<0) cout<<"mutate "<<param->ch[i].f<<endl;
 }
 }
 param->avg += param->ch[i].f;
 //cout<<"compare: "<<param.best<<" "<<param.ch[i].f<<endl;
 if(param->ch[i].f>param->best)
 {
 //cout<<i<<" "<<"before best "<<param->best<<endl;
 param->best=param->ch[i].f;
 //cout<<i<<" "<<"after best "<<param->best<<endl;
 }
 }
 
 
 
 */











