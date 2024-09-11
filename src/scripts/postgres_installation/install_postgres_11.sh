#!/usr/bin/env bash

set -e
set -x

if [[ -e FLAG_INSTALL_DONE ]]
then
    echo "skip installation"
else

  wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
  echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/postgresql-pgdg.list > /dev/null
  sudo apt update
  sudo apt install -y postgresql-11
  sudo apt install -y postgresql-server-dev-11
  sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'bM2YGRAX*bG_QAilUid§2iD';"
  sudo service postgresql restart
  sudo apt install gcc
  sudo apt install make
  wget https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/REL11_1_3_9.tar.gz
  tar xzvf REL11_1_3_9.tar.gz
  cd pg_hint_plan-REL11_1_3_9
  make
  sudo make install
  cd ..
  rm REL11_1_3_9.tar.gz
  sudo service postgresql restart
  sudo cp cost-eval/src/conf/postgres/modified-postgresql11.conf /etc/postgresql/11/main/postgresql.conf # ToDo!?
  sudo cp cost-eval/src/conf/postgres/pg_hba.conf /etc/postgresql/11/main/pg_hba.conf
  sudo service postgresql restart
  touch FLAG_INSTALL_DONE
fi