#!/usr/bin/env bash
wget -P /tmp http://coppeliarobotics.com/files/V-REP_PRO_EDU_V3_5_0_Linux.tar.gz
cd /tmp
tar -xzvf V-REP_PRO_EDU_V3_5_0_Linux
sudo mv V-REP_PRO_EDU_V3_5_0_Linux /opt/V-REP
rm -rf V-REP_PRO_EDU_V3_5_0_Linux
echo "export V_REP=/opt/V-REP" >> ~/.bashrc
source ~/.bashrc
