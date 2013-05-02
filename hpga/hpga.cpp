// a fast genetic algorithm for the 0-1 knapsack problem
// test case: 10000 items, 50 knapsack size
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

pthread_mutex_t island_lock[2];
pthread_barrier_t barrier;

clock_t start;

long st = time(0);

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
    
    
	int limit; // knapsack weight limit
	int pop; // chromosome population size
	int gens; // maximum number of generations
	int disc; // chromosomes discarded via elitism
	int dimw;
    
	vector<chromo> ch;
	vector<int> w,v;
    
	double *avg;
	int *best;
	int *best_index;
	int *worst;
	int *worst_index;
	int *p;
    
};

Param *param = new Param;
int inner_num_thread = 2;

int fitness(bool*& x, const int dimc, const vector<int>& v, const vector<int>& w, const int limit) {
	int fit = 0, wsum = 0;
	for(int i=0;i<dimc;i++) {
		wsum += x[i]*w[i];
		fit += x[i]*v[i];
	}
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

struct ThreadInfo {
	int index;
	int begin;
	int end;
};

void inner_threadFunction(ThreadInfo *info) {
	
	int index = info->index;
	int begin = info->begin;
	int end = info->end;
	int dimw = (int)param->dimw;

		for(int i=begin;i<end;i++)
		{
			if(i>param->pop-param->disc) { // elitism - only processes the discarded chromosomes
				if(coin(param->crp)==1) { // crossover section
					param->ind = param->parc+round(10*RND); // choosing parents for crossover
					param->ind2 = param->parc+1+round(10*RND);
					// choose a crossover strategy here
					crossover1p(param->ch[param->ind%param->pop],param->ch[param->ind2%param->pop],param->ch[i],dimw,round(RND*(dimw-1)));
					//crossoverrand(ch[ind],ch[ind2],ch[i],dimw);
					//crossoverarit(ch[0],ch[1],ch[i],dimw);
					param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, param->limit);
					param->parc += 1;
				}
				else { // mutation section
					param->ch[i].mutate(dimw,1);
					param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, param->limit);
				}
			}
			
        		//pthread_mutex_lock(&island_lock[index]);

			param->avg[index] += param->ch[i].f;

			if (param->ch[i].f > param->best[index]) {
				param->best[index] = param->ch[i].f;  
				param->best_index[index] = i;  
			}

			if (param->ch[i].f < param->worst[index]) {
				param->worst[index] = param->ch[i].f;    
				param->worst_index[index] = i;  
			}
			
        		//pthread_mutex_unlock(&island_lock[index]);
		}
}

void mig_threadFunction(int *i)
{
	//add code
	pthread_t *thread = new pthread_t[inner_num_thread];
	ThreadInfo info[inner_num_thread];

	int size = param->pop/param->number_of_thread;
	int index = (int) i;
	int begin = index*size;
	int end = begin+size-1;
	int dimw = (int)param->dimw;
	param->p[index] = 0;
	
	size = (end-begin)/inner_num_thread;
	info[0].begin = begin;
 	info[0].end = begin+size-1;
	info[0].index = index;
	for (int i = 1; i<inner_num_thread; i++)
	{
		info[i].index = index;
		info[i].begin = info[i-1].begin + size;
		info[i].end = info[i-1].end + size;
	}
	

	for(int p=0;p<100000;p++)
	{
        	param->p[index]=p;

		param->best[index] = 0;
		param->worst[index] = INT_MAX;
		param->avg[index] = 0.0;

		
        
		std::sort(&param->ch[begin], &param->ch[end], cfit);
		

		for (int j=0; j<inner_num_thread; j++)
		{
			pthread_create(&thread[j], NULL, (void *(*)(void *))inner_threadFunction, (void*)&info[j]);
		}

		for (int j=0; j<inner_num_thread; j++)
		{
			pthread_join(thread[j], NULL);
		}


		param->parc = 0;
	
		if(p%5==0) {
			printf("#%d:%d\t",p, index);
			printf("best fitness: %d \t",param->best[index]);
			printf("avg fitness: %f",param->avg[index]/size);
			printf("\n");
			//if(p >= 1000) {
			if(param->best[index]>=3860){
				clock_t end = clock();
				long ter = time(0);
				double t = (double)(ter-st);//(end-start)/CLOCKS_PER_SEC;
				printf("\nCompletion time: %fs.\n",t);
				break;			
			}
		}
		
		//Migration
		int migration_method = 2;

		//Migration 1: migrate the best one
		if(p%2==0 && 1==migration_method) {
			
    			pthread_barrier_wait(&barrier);
			//let thread zero handle the migration
			if(index == 0) {
				int thread_of_best = 0;
				for(int i=0; i<param->number_of_thread; i++) {
					if(param->best[i] > param->best[thread_of_best]){
						thread_of_best = i;
					}				
				}

				for(int i=0; i<param->number_of_thread; i++){
					if(param->best[thread_of_best] > param->worst[i]){
						int thread_worst_ch_index = param->worst_index[i];
						int thread_best_ch_index = param->best_index[thread_of_best];
						for(int j=0;j<dimw;j++) {
								param->ch[thread_worst_ch_index].items[j] = param->ch[thread_best_ch_index].items[j];
						}	
						param->ch[thread_worst_ch_index].f = param->ch[thread_best_ch_index].f;	
					}				
				}
			}
    			pthread_barrier_wait(&barrier);
		}

		//Migration 2: migrate the best ones by rotation. 1->2;
		double migration_rate = 0.9;

		if(p%150==0) {
			int p_max=0;
			for(int k=0; k<param->number_of_thread; k++)
				if(param->p[k] > p_max)
					p_max = param->p[k];

			if(param->p[index] == p_max)				 
				pthread_barrier_wait(&barrier);
		}		
		
		if(p%1==0 && 2==migration_method) {
    			//pthread_barrier_wait(&barrier);
			int target = (index+1) % param->number_of_thread;

			int target_end = (target+1)*size - 1;
			int target_begin =  target_end - (int)(migration_rate*size);
			
			if(1) {
				int k=begin;
				for(int i=target_begin; i<target_end; i++){
					if(param->ch[k].f > param->ch[i].f)
					{
						for(int j=0;j<dimw;j++) {
							param->ch[i].items[j] = param->ch[k].items[j];		
						}

						param->ch[i].f = param->ch[k].f;
					}
					k++;	
				}
			}
    			//pthread_barrier_wait(&barrier);
		}
	}
}


int main(int argc, const char* argv[])
{
    
	if (argc<2)
	{
		cout<<"Invalid Arguments"<<endl;
		exit(-1);
	}
    
    
	printf("Start \n");
	param->number_of_thread = 2;
    	inner_num_thread = atoi(argv[1]);
	param->best = new int[param->number_of_thread];
	param->worst = new int[param->number_of_thread];
	param->avg = new double[param->number_of_thread];
	param->best_index = new int[param->number_of_thread];
	param->worst_index = new int[param->number_of_thread];
	param->p = new int[param->number_of_thread];
	
	pthread_t *thread = new pthread_t[param->number_of_thread];
	    
	pthread_mutex_init(&lock, NULL);
	for(int i=0;i<2;i++){
		pthread_mutex_init(&island_lock[i], NULL);
	}
	pthread_barrier_init(&barrier, NULL, param->number_of_thread);
    
	
	srand(time(NULL));
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
	const int pop = 108; // chromosome population size
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
	param->ind = ind;
	param->ind2 = ind2;
	param->parc = parc;
    
	double crp = 0.35; // crossover probability
	param->crp = crp;
    
	//vector<chromo> ch(pop,chromo(dimw));
	param->ch = vector<chromo>(pop,chromo(dimw));
    
	bool **c = new bool*[pop];
	for(int i=0;i<pop;i++) c[i] = new bool[dimw];
    
	start = clock();
	st = time(0);

	printf("Initializing population with a greedy algorithm...");
	initpopg(c,param->w,param->v,dimw,limit,pop);
    
	printf("done!");
    
	for(int i=0;i<pop;i++) {
		param->ch[i].items = c[i];
		param->ch[i].f = fitness(param->ch[i].items, dimw ,param->v, param->w, limit);
	}
    
	printf("\n\n");
    	
	for (int i = 0; i<param->number_of_thread; i++)
	{
		pthread_create(&thread[i], NULL, (void *(*)(void *))mig_threadFunction, (void*)i);
	}

	best = -1;
	for (int i = 0; i<param->number_of_thread; i++)
	{
		pthread_join(thread[i], NULL);
	}
    
	for (int i = 0; i<param->number_of_thread; i++)
	{
		cout<<"thread: "<<i<<" "<<param->best[i]<<endl;
		if (best<param->best[i])
			best = param->best[i];
	}
    
	cout<<"best "<<best<<endl;
    
end:
	printf("\n\n");
	clock_t end = clock();
	long ter = time(0);
	double t = (double)(ter-st);//(end-start)/CLOCKS_PER_SEC;
	printf("\nCompletion time: %fs.\n",t);

	free(param->best);
	free(param->worst);
	free(param->avg);
	free(param->best_index);
	free(param->worst_index);
	free(param->p);

    	pthread_barrier_destroy(&barrier);
	return 0;
}











