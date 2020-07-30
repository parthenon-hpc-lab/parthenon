#!/bin/bash

LCOV=$1
GCOV=$2
WORKING_PATH=$3
if [ -d "${WORKING_PATH}/CombinedCoverage" ]; then rm -Rf ${WORKING_PATH}/CombinedCoverage; fi

items=$(ls ${WORKING_PATH})
COMMAND_ARGS="--gcov-tool ${GCOV}"
for item in ${items[@]}
do
  if [ -d "$item" ]
  then
    COMMAND_ARGS=${COMMAND_ARGS}" -a ${WORKING_PATH}/${item}/report.all"
  fi
done

mkdir -p ${WORKING_PATH}/CombinedCoverage
echo "COMMAND args "$COMMAND_ARGS
command ${LCOV} $COMMAND_ARGS --output-file ${WORKING_PATH}/CombinedCoverage/report.combined.all
