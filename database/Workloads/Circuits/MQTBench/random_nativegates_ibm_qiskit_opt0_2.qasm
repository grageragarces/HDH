// Benchmark was created by MQT Bench on 2024-03-18
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2
// Used Gate Set: ['id', 'rz', 'sx', 'x', 'cx', 'measure', 'barrier']

OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg meas[2];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[0],q[1];
rz(1.304903297657757) q[1];
cx q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
cx q[1],q[0];
rz(3.2220464314480877) q[0];
cx q[1],q[0];
rz(-3*pi/2) q[0];
sx q[0];
rz(2.2852169485325513) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(2*pi) q[1];
sx q[1];
rz(7*pi/2) q[1];
cx q[1],q[0];
rz(0) q[0];
sx q[0];
rz(0.13532095111165665) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[1];
sx q[1];
rz(6.147864356067929) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[1],q[0];
rz(-2.6682620430456936) q[0];
rz(-3*pi/2) q[1];
sx q[1];
rz(-pi/2) q[1];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];