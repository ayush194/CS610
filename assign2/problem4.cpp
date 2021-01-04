/**
 * Author: Ayush Kumar
 * Roll No: 170195
 */

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <queue>

int n, shared_array[16], customers_served = 0;
std::queue<int> token_queue;
pthread_mutex_t token_queue_lock;
pthread_mutex_t print_lock;

typedef struct CustomerThreadInfo {
	pthread_t id;
	pthread_mutex_t lock;
	pthread_cond_t cond;
	int teller_id;
	bool is_write_allowed;
	bool is_read_allowed;
} CustomerThreadInfo;

typedef struct TellerThreadInfo {
	pthread_t id;
} TellerThreadInfo;

CustomerThreadInfo customer_thread[24];
TellerThreadInfo teller_thread[2];
// customer_thread[i] represents the thread for the customer with token i
// teller_thread[i] represents the teller i

void* handleCustomer(void* arg) {
	int token = *((int*)arg);
	free(arg);
	pthread_mutex_lock(&token_queue_lock);
	printf("Token %d\n", token);
	token_queue.push(token);
	customers_served++;
	pthread_mutex_unlock(&token_queue_lock);
	// wait for its turn
	pthread_mutex_lock(&customer_thread[token].lock);
	while (!customer_thread[token].is_write_allowed) 
		pthread_cond_wait(&customer_thread[token].cond, &customer_thread[token].lock);
	// printf("Token %d: Customer writes\n", token);
	for (int i = customer_thread[token].teller_id*8; i < customer_thread[token].teller_id*8+8; i++) 
		shared_array[i] = token;
	customer_thread[token].is_read_allowed = true;
	pthread_mutex_unlock(&customer_thread[token].lock);
	pthread_cond_signal(&customer_thread[token].cond);
	return NULL;
}

void* handleTeller(void*) {
	pthread_t curr_thread_id = pthread_self();
	int teller_id = curr_thread_id == teller_thread[0].id ? 0 : 1;
	while(true) {
		// teller is free, get the next element to be processed
		pthread_mutex_lock(&token_queue_lock);
		if (token_queue.empty()) {
			if (customers_served == n) {pthread_mutex_unlock(&token_queue_lock); break;} // no more customers to process
			else {pthread_mutex_unlock(&token_queue_lock); continue;}
		}
		int next_token = token_queue.front();
		token_queue.pop();
		pthread_mutex_unlock(&token_queue_lock);
		// process the customer having token next_token, i.e. thread customer_thread_id[next_token]
		customer_thread[next_token].teller_id = teller_id;
		customer_thread[next_token].is_write_allowed = true;
		pthread_cond_signal(&customer_thread[next_token].cond);
		pthread_mutex_lock(&customer_thread[next_token].lock);
		while (!customer_thread[next_token].is_read_allowed) {
			pthread_cond_wait(&customer_thread[next_token].cond, &customer_thread[next_token].lock);
		}
		// printf("Token %d: Teller reads\n", next_token);
		pthread_mutex_lock(&print_lock);
		for (int i = teller_id*8; i < teller_id*8+8; i++) {
			printf("%d", shared_array[i]);
			if (i != teller_id*8+7) printf(" ");
		}
		printf("\n");
		pthread_mutex_unlock(&print_lock);
		pthread_mutex_unlock(&customer_thread[next_token].lock);
		sleep(5);
	}
	printf("Teller %d has finished serving customers!\n", teller_id);
	return NULL;
}


int main() {
	printf("Enter N: ");
	scanf("%d", &n);
	if (n > 24) {
		printf("This bank cannot handle more than 24 customers at this time.\n");
		exit(-1);
	}
	pthread_mutex_init(&token_queue_lock, NULL);
	pthread_mutex_init(&print_lock, NULL);
	for (int i = 0; i < n; i++) {
		pthread_mutex_init(&customer_thread[i].lock, NULL);
		pthread_cond_init(&customer_thread[i].cond, NULL);
		customer_thread[i].is_write_allowed = customer_thread[i].is_read_allowed = false;
		// give this customer token i and spawn a thread for him/her
		int* arg = (int*)malloc(sizeof(int)); *arg = i;
		if (pthread_create(&customer_thread[i].id, NULL, handleCustomer, (void*)arg) != 0) {
			perror("pthread_create_customer");
			exit(-1);
		}
	}
	for (int i = 0; i < 2; i++) {
		// create the thread for teller i
		if (pthread_create(&teller_thread[i].id, NULL, handleTeller, NULL) != 0) {
			perror("pthread_create_teller");
			exit(-1);
		}
	}
	// wait for all customer threads to finish execution
	for (int i = 0; i < n; i++) {
		pthread_join(customer_thread[i].id, NULL);
	}
	// wait for all teller threads to finish execution
	for (int i = 0; i < 2; i++) {
		pthread_join(teller_thread[i].id, NULL);
	}
	return 0;
}