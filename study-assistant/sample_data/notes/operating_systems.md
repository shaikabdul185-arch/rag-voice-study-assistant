# Operating Systems — Lecture Notes

## Processes and Threads
A process is a program in execution with its own address space. Threads share the
address space of their process but have their own stack and registers. Context
switching between threads of the same process is cheaper than between processes
because the page tables do not change.

---

## Deadlock
Four conditions must all hold for deadlock (Coffman conditions):
1. Mutual exclusion — a resource can be held by only one process.
2. Hold and wait — a process holds resources while waiting for others.
3. No preemption — resources cannot be forcibly taken away.
4. Circular wait — a cycle of processes each waiting on the next.

Breaking any one condition prevents deadlock. A common technique is to impose a global
ordering on locks to prevent circular wait. The Banker's algorithm avoids deadlock by
only granting requests that leave the system in a safe state.

---

## CPU Scheduling
First-Come First-Served (FCFS) is simple but suffers from the convoy effect.
Shortest Job First (SJF) minimises average waiting time but needs burst-length
estimates. Round Robin (RR) gives each process a time quantum q; small q improves
response time but increases context-switch overhead.

Average waiting time = (sum of waiting times) / (number of processes), where
waiting time = turnaround time - burst time.
