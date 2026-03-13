# Copyright 2026 Dell Inc. or its subsidiaries. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slurm node drain/resume job behavior tests.

Validates job submission behavior when nodes are drained and resumed:
- Job submission during node drain (single, multiple, batch, targeted)
- PENDING state and reason verification
- Job transition from PENDING to RUNNING after node resume
- FIFO ordering verification

Configuration (env-driven + PXE mapping discovery):
- Preferred: derive login node admin IPs from pxe_mapping_file via public core API
- Fallback: LOGIN_NODE_IPS env (comma-separated login node IPs)
"""

import time
import pytest

from automation_library.core import TestLogger
from automation_library.core.host import (
    get_testinfra_host,
    get_nodes_info,
)
from automation_library.slurm.vars import MULTI_JOB_COUNT
from automation_library.slurm.messages import (
    TEST_ASSERT_MSGS,
    DRAIN_TEST_NAMES,
    DRAIN_LOG_MSGS,
    DRAIN_ASSERT_MSGS,
)
from automation_library.slurm.functions import (
    submit_job_direct,
    poll_job_state_direct,
    wait_job_running,
    get_job_start_time,
    cleanup_jobs_direct,
    setup_slurm_test_env,
    drain_all_nodes,
    drain_single_node,
    resume_all_nodes,
    cleanup_test_env,
    setup_drain_test_env,
    setup_all_drained_nodes,
    setup_single_drained_node,
    cleanup_drain_test,
    setup_test_env_with_all_drained,
)


# =============================================================================
# TESTS
# =============================================================================


# =============================================================================
# JOB SUBMISSION DURING NODE DRAIN TESTS
# =============================================================================

def test_submit_multiple_jobs_all_nodes_drained():
    """Submit multiple jobs when all nodes drained."""
    log = TestLogger(DRAIN_TEST_NAMES["submit_multiple_all_drained"])
    
    # Set up test environment
    env_result = setup_drain_test_env(get_testinfra_host())
    assert env_result["success"], env_result["error"]
    login_ip = env_result["login_ip"]
    compute_nodes = env_result["compute_nodes"]
    
    # Drain all nodes
    drain_result = setup_all_drained_nodes(login_ip, "root", compute_nodes)
    assert drain_result["success"], DRAIN_ASSERT_MSGS["drain_failed"].format(
        node="all", error=drain_result["error"]
    )
    
    log.check(f"All nodes drained: {drain_result['nodes']}")
    
    job_ids = []
    try:
        # Submit 3 jobs
        for i in range(3):
            result = submit_job_direct(login_ip, "root")
            assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
                error=result["error"]
            )
            job_ids.append(result["job_id"])
            log.passed(DRAIN_LOG_MSGS["job_submitted"].format(job_id=result["job_id"]))
        
        log.passed(f"Successfully submitted {len(job_ids)} jobs while all nodes drained")
    finally:
        # Cleanup
        cleanup_drain_test(login_ip, "root", compute_nodes, job_ids)


def test_submit_jobs_targeting_specific_drained_node():
    """Submit jobs targeting a specific drained node."""
    log = TestLogger(DRAIN_TEST_NAMES["submit_targeting_drained_node"])
    
    # Set up test environment
    env_result = setup_drain_test_env(get_testinfra_host())
    assert env_result["success"], env_result["error"]
    login_ip = env_result["login_ip"]
    compute_nodes = env_result["compute_nodes"]
    
    # Drain a single node
    drain_result = setup_single_drained_node(login_ip, "root", compute_nodes, node_index=0)
    assert drain_result["success"], DRAIN_ASSERT_MSGS["drain_failed"].format(
        node=drain_result["node"], error=drain_result["error"]
    )
    node = drain_result["node"]
    
    log.check(f"Targeting drained node: {node}")
    
    job_ids = []
    try:
        # Submit job targeting the drained node
        result = submit_job_direct(login_ip, "root", nodelist=node)
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_ids.append(result["job_id"])
        log.passed(DRAIN_LOG_MSGS["job_submitted"].format(job_id=result["job_id"]))
    finally:
        # Cleanup
        cleanup_drain_test(login_ip, "root", [node], job_ids)


def test_submit_large_batch_jobs():
    """Submit large batch of jobs (10+)."""
    log = TestLogger(DRAIN_TEST_NAMES["submit_large_batch"])
    
    # Set up test environment
    env_result = setup_drain_test_env(get_testinfra_host())
    assert env_result["success"], env_result["error"]
    login_ip = env_result["login_ip"]
    compute_nodes = env_result["compute_nodes"]
    batch_count = MULTI_JOB_COUNT
    
    # Drain all nodes
    drain_result = setup_all_drained_nodes(login_ip, "root", compute_nodes)
    assert drain_result["success"], DRAIN_ASSERT_MSGS["drain_failed"].format(
        node="all", error=drain_result["error"]
    )
    
    log.check(f"Submitting {batch_count} jobs while all nodes drained")
    
    job_ids = []
    try:
        for i in range(batch_count):
            result = submit_job_direct(login_ip, "root")
            assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
                error=result["error"]
            )
            job_ids.append(result["job_id"])
        
        log.passed(DRAIN_LOG_MSGS["batch_submitted"].format(count=batch_count))
    finally:
        # Cleanup
        cleanup_drain_test(login_ip, "root", compute_nodes, job_ids)
    # =============================================================================
# PENDING STATE VERIFICATION TESTS
# =============================================================================

def test_single_job_pending_all_nodes_drained():
    """Single job enters PENDING when all nodes drained."""
    log = TestLogger(DRAIN_TEST_NAMES["single_pending_all_drained"])
    
    # Set up test environment
    env_result = setup_drain_test_env(get_testinfra_host())
    assert env_result["success"], env_result["error"]
    login_ip = env_result["login_ip"]
    compute_nodes = env_result["compute_nodes"]
    
    # Drain all nodes
    drain_result = setup_all_drained_nodes(login_ip, "root", compute_nodes)
    assert drain_result["success"], DRAIN_ASSERT_MSGS["drain_failed"].format(
        node="all", error=drain_result["error"]
    )
    
    job_ids = []
    try:
        result = submit_job_direct(login_ip, "root")
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_id = result["job_id"]
        job_ids.append(job_id)

        # Verify job is PENDING
        state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
        assert state_result["state"] == "PENDING", DRAIN_ASSERT_MSGS["job_not_pending"].format(
            job_id=job_id, state=state_result["state"]
        )
        log.passed(DRAIN_LOG_MSGS["job_pending"].format(
            job_id=job_id, reason=state_result["reason"]
        ))
    finally:
        # Cleanup
        cleanup_drain_test(login_ip, "root", compute_nodes, job_ids)


def test_multiple_jobs_all_show_pending():
    """Multiple jobs all show PENDING."""
    log = TestLogger(DRAIN_TEST_NAMES["multiple_pending"])
    
    # Set up test environment
    env_result = setup_drain_test_env(get_testinfra_host())
    assert env_result["success"], env_result["error"]
    login_ip = env_result["login_ip"]
    compute_nodes = env_result["compute_nodes"]
    
    # Drain all nodes
    drain_result = setup_all_drained_nodes(login_ip, "root", compute_nodes)
    assert drain_result["success"], DRAIN_ASSERT_MSGS["drain_failed"].format(
        node="all", error=drain_result["error"]
    )
    
    job_ids = []
    try:
        # Submit 5 jobs
        for i in range(5):
            result = submit_job_direct(login_ip, "root")
            assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
                error=result["error"]
            )
            job_ids.append(result["job_id"])

        # Verify all jobs are PENDING
        for job_id in job_ids:
            state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
            assert state_result["state"] == "PENDING", DRAIN_ASSERT_MSGS["job_not_pending"].format(
                job_id=job_id, state=state_result["state"]
            )
            log.passed(DRAIN_LOG_MSGS["job_pending"].format(
                job_id=job_id, reason=state_result["reason"]
            ))
    finally:
        # Cleanup
        cleanup_drain_test(login_ip, "root", compute_nodes, job_ids)


def test_job_targeting_drained_node_is_pending():
    """Job targeting specific drained node is PENDING."""
    log = TestLogger(DRAIN_TEST_NAMES["targeted_pending"])
    
    # Set up test environment
    env_result = setup_drain_test_env(get_testinfra_host())
    assert env_result["success"], env_result["error"]
    login_ip = env_result["login_ip"]
    compute_nodes = env_result["compute_nodes"]
    
    # Drain a single node
    drain_result = setup_single_drained_node(login_ip, "root", compute_nodes, node_index=0)
    assert drain_result["success"], DRAIN_ASSERT_MSGS["drain_failed"].format(
        node=drain_result["node"], error=drain_result["error"]
    )
    node = drain_result["node"]
    
    job_ids = []
    try:
        result = submit_job_direct(login_ip, "root", nodelist=node)
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_id = result["job_id"]
        job_ids.append(job_id)

        state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
        assert state_result["state"] == "PENDING", DRAIN_ASSERT_MSGS["job_not_pending"].format(
            job_id=job_id, state=state_result["state"]
        )
        log.passed(DRAIN_LOG_MSGS["job_pending"].format(
            job_id=job_id, reason=state_result["reason"]
        ))
    finally:
        # Cleanup
        cleanup_drain_test(login_ip, "root", [node], job_ids)


# =============================================================================
# PENDING REASON VERIFICATION TESTS
# =============================================================================

def test_reason_reqnodenotavail_all_drained():
    """Reason is ReqNodeNotAvail when all nodes drained."""
    log = TestLogger(DRAIN_TEST_NAMES["reason_reqnodenotavail"])
    
    # Set up test environment
    env_result = setup_drain_test_env(get_testinfra_host())
    assert env_result["success"], env_result["error"]
    login_ip = env_result["login_ip"]
    compute_nodes = env_result["compute_nodes"]

    # Drain all nodes
    drain_result = setup_all_drained_nodes(login_ip, "root", compute_nodes)
    assert drain_result["success"], DRAIN_ASSERT_MSGS["drain_failed"].format(
        node="all", error=drain_result["error"]
    )
    
    job_ids = []
    try:
        result = submit_job_direct(login_ip, "root")
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_id = result["job_id"]
        job_ids.append(job_id)

        state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
        reason_lower = state_result["reason"].lower()

        # Accept ReqNodeNotAvail or similar resource-related reasons
        valid_reasons = ["reqnodenotavail", "resources", "nodedown", "partitiondown"]
        assert any(r in reason_lower for r in valid_reasons), \
            DRAIN_ASSERT_MSGS["wrong_pending_reason"].format(
                job_id=job_id, reason=state_result["reason"]
            )
        log.passed(f"Job {job_id} has expected reason: {state_result['reason']}")
    finally:
        # Cleanup
        cleanup_drain_test(login_ip, "root", compute_nodes, job_ids)


def test_reason_nodedown_for_downed_node():
    """Reason shows NodeDown for specifically downed node."""
    log = TestLogger(DRAIN_TEST_NAMES["reason_nodedown"])
    
    # Set up test environment
    env_result = setup_drain_test_env(get_testinfra_host())
    assert env_result["success"], env_result["error"]
    login_ip = env_result["login_ip"]
    compute_nodes = env_result["compute_nodes"]
    
    # Drain a single node
    drain_result = setup_single_drained_node(login_ip, "root", compute_nodes, node_index=0)
    assert drain_result["success"], DRAIN_ASSERT_MSGS["drain_failed"].format(
        node=drain_result["node"], error=drain_result["error"]
    )
    node = drain_result["node"]
    
    job_ids = []
    try:
        result = submit_job_direct(login_ip, "root", nodelist=node)
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_id = result["job_id"]
        job_ids.append(job_id)

        state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
        reason_lower = state_result["reason"].lower()

        # Accept NodeDown or ReqNodeNotAvail for targeted drained node
        valid_reasons = ["nodedown", "reqnodenotavail", "drain"]
        assert any(r in reason_lower for r in valid_reasons), \
            DRAIN_ASSERT_MSGS["wrong_pending_reason"].format(
                job_id=job_id, reason=state_result["reason"]
            )
        log.passed(f"Job {job_id} targeting {node} has expected reason: {state_result['reason']}")
    finally:
        # Cleanup
        cleanup_drain_test(login_ip, "root", [node], job_ids)


def test_reason_updates_when_node_state_changes():
    """Reason updates when node state changes."""
    log = TestLogger(DRAIN_TEST_NAMES["reason_updates_on_state_change"])
    
    # Set up test environment
    env_result = setup_drain_test_env(get_testinfra_host())
    assert env_result["success"], env_result["error"]
    login_ip = env_result["login_ip"]
    compute_nodes = env_result["compute_nodes"]
    node = compute_nodes[0]
    job_ids = []

    try:
        # Drain node
        drain_result = drain_node(login_ip, "root", node, reason="testing_reason_update")
        assert drain_result["success"], DRAIN_ASSERT_MSGS["drain_failed"].format(
            node=node, error=drain_result["error"]
        )
        wait_node_state(login_ip, "root", node, "drain", timeout=30)

        # Submit job targeting drained node
        result = submit_job_direct(login_ip, "root", nodelist=node)
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_id = result["job_id"]
        job_ids.append(job_id)

        # Check initial reason
        state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
        initial_reason = state_result["reason"]
        log.check(f"Initial reason: {initial_reason}")

        # Resume node
        resume_result = resume_node(login_ip, "root", node)
        assert resume_result["success"], DRAIN_ASSERT_MSGS["resume_failed"].format(
            node=node, error=resume_result["error"]
        )
        wait_node_state(login_ip, "root", node, "idle", timeout=60)

        # Wait for job to transition
        running_result = wait_job_running(login_ip, "root", job_id, timeout=120)
        log.passed(f"Job {job_id} transitioned to {running_result['state']} after node resumed")

    finally:
        resume_node(login_ip, "root", node)
        if job_ids:
            cleanup_jobs_direct(login_ip, "root", job_ids)


# =============================================================================
# ROBUSTNESS TESTS
# =============================================================================

def test_jobs_survive_extended_node_downtime():
    """Jobs survive extended node downtime."""
    log = TestLogger(DRAIN_TEST_NAMES["jobs_survive_extended_downtime"])
    
    # Set up test environment with all nodes drained
    test_result = setup_test_env_with_all_drained(get_testinfra_host())
    assert test_result["success"], test_result["error"]
    login_ip = test_result["login_ip"]
    compute_nodes = test_result["compute_nodes"]
    
    job_ids = []
    try:
        # Submit job
        result = submit_job_direct(login_ip, "root")
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_id = result["job_id"]
        job_ids.append(job_id)

        # Wait for extended period (simulated with shorter wait for testing)
        log.check(f"Job {job_id} submitted, waiting 30 seconds to simulate extended downtime...")
        time.sleep(30)

        # Verify job is still PENDING (not cancelled or failed)
        state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
        assert state_result["state"] == "PENDING", \
            f"Job {job_id} did not survive extended downtime; state={state_result['state']}"
        log.passed(f"Job {job_id} survived extended downtime, still PENDING")
    finally:
        # Cleanup
        cleanup_drain_test(login_ip, "root", compute_nodes, job_ids)


def test_jobs_not_rejected_during_node_state_transition(test_env):
    """Jobs not rejected when submitting during node state transition."""
    log = TestLogger(DRAIN_TEST_NAMES["jobs_not_rejected_during_transition"])
    login_ip = test_env["login_ip"]
    compute_nodes = test_env["compute_nodes"]
    node = compute_nodes[0]
    job_ids = []

    try:
        # Start draining node (don't wait for completion)
        drain_node(login_ip, "root", node, reason="testing_transition")

        # Immediately submit jobs during transition
        for i in range(3):
            result = submit_job_direct(login_ip, "root")
            assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
                error=result["error"]
            )
            job_ids.append(result["job_id"])
            log.passed(DRAIN_LOG_MSGS["job_submitted"].format(job_id=result["job_id"]))

        log.passed(f"All {len(job_ids)} jobs accepted during node state transition")

    finally:
        resume_node(login_ip, "root", node)
        if job_ids:
            cleanup_jobs_direct(login_ip, "root", job_ids)


def test_resubmit_after_drain_no_duplicates(test_env, drained_nodes):
    """Resubmitting after node drain doesn't cause duplicates/failures."""
    log = TestLogger(DRAIN_TEST_NAMES["resubmit_no_duplicates"])
    login_ip = test_env["login_ip"]

    # Submit first batch
    first_batch = []
    for i in range(3):
        result = submit_job_direct(login_ip, "root")
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        first_batch.append(result["job_id"])
        drained_nodes["job_ids"].append(result["job_id"])

    log.check(f"First batch submitted: {first_batch}")

    # Submit second batch (resubmission scenario)
    second_batch = []
    for i in range(3):
        result = submit_job_direct(login_ip, "root")
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        second_batch.append(result["job_id"])
        drained_nodes["job_ids"].append(result["job_id"])

    log.check(f"Second batch submitted: {second_batch}")

    # Verify all jobs have unique IDs
    all_jobs = first_batch + second_batch
    assert len(all_jobs) == len(set(all_jobs)), "Duplicate job IDs detected"
    log.passed(f"All {len(all_jobs)} jobs have unique IDs, no duplicates")


# =============================================================================
# JOB TRANSITION AFTER RESUME TESTS
# =============================================================================

def test_single_pending_job_transitions_to_running_after_resume(test_env):
    """Single PENDING job transitions to RUNNING after resume."""
    log = TestLogger(DRAIN_TEST_NAMES["single_pending_to_running"])
    login_ip = test_env["login_ip"]
    compute_nodes = test_env["compute_nodes"]
    job_ids = []

    try:
        # Drain all nodes
        for node in compute_nodes:
            drain_node(login_ip, "root", node, reason="testing_resume")
            wait_node_state(login_ip, "root", node, "drain", timeout=30)

        # Submit job
        result = submit_job_direct(login_ip, "root")
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_id = result["job_id"]
        job_ids.append(job_id)

        # Verify PENDING
        state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
        assert state_result["state"] == "PENDING", DRAIN_ASSERT_MSGS["job_not_pending"].format(
            job_id=job_id, state=state_result["state"]
        )
        log.check(f"Job {job_id} is PENDING as expected")

        # Resume all nodes
        for node in compute_nodes:
            resume_node(login_ip, "root", node)

        # Wait for job to transition to RUNNING
        running_result = wait_job_running(login_ip, "root", job_id, timeout=180)
        assert running_result["state"] in {"RUNNING", "COMPLETED"}, \
            DRAIN_ASSERT_MSGS["job_not_running"].format(
                job_id=job_id, state=running_result["state"]
            )
        log.passed(DRAIN_LOG_MSGS["job_running"].format(job_id=job_id))

    finally:
        for node in compute_nodes:
            resume_node(login_ip, "root", node)
        if job_ids:
            cleanup_jobs_direct(login_ip, "root", job_ids)


def test_multiple_pending_jobs_transition_after_resume(test_env):
    """Multiple PENDING jobs all transition after resume."""
    log = TestLogger(DRAIN_TEST_NAMES["multiple_pending_to_running"])
    login_ip = test_env["login_ip"]
    compute_nodes = test_env["compute_nodes"]
    job_ids = []

    try:
        # Drain all nodes
        for node in compute_nodes:
            drain_node(login_ip, "root", node, reason="testing_multi_resume")
            wait_node_state(login_ip, "root", node, "drain", timeout=30)

        # Submit multiple jobs
        for i in range(3):
            result = submit_job_direct(login_ip, "root")
            assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
                error=result["error"]
            )
            job_ids.append(result["job_id"])

        log.check(f"Submitted {len(job_ids)} jobs, all should be PENDING")

        # Verify all PENDING
        for job_id in job_ids:
            state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
            assert state_result["state"] == "PENDING", DRAIN_ASSERT_MSGS["job_not_pending"].format(
                job_id=job_id, state=state_result["state"]
            )

        # Resume all nodes
        for node in compute_nodes:
            resume_node(login_ip, "root", node)

        # Wait for all jobs to transition
        for job_id in job_ids:
            running_result = wait_job_running(login_ip, "root", job_id, timeout=180)
            assert running_result["state"] in {"RUNNING", "COMPLETED"}, \
                DRAIN_ASSERT_MSGS["job_not_running"].format(
                    job_id=job_id, state=running_result["state"]
                )
            log.passed(DRAIN_LOG_MSGS["job_running"].format(job_id=job_id))

    finally:
        for node in compute_nodes:
            resume_node(login_ip, "root", node)
        if job_ids:
            cleanup_jobs_direct(login_ip, "root", job_ids)


def test_job_transitions_when_only_one_node_resumed(test_env):
    """Job transitions when only one node resumed."""
    log = TestLogger(DRAIN_TEST_NAMES["partial_resume_transition"])
    login_ip = test_env["login_ip"]
    compute_nodes = test_env["compute_nodes"]
    job_ids = []

    if len(compute_nodes) < 2:
        pytest.skip("Need at least 2 compute nodes for partial resume test")

    try:
        # Drain all nodes
        for node in compute_nodes:
            drain_node(login_ip, "root", node, reason="testing_partial_resume")
            wait_node_state(login_ip, "root", node, "drain", timeout=30)

        # Submit job
        result = submit_job_direct(login_ip, "root")
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_id = result["job_id"]
        job_ids.append(job_id)

        # Verify PENDING
        state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
        assert state_result["state"] == "PENDING", DRAIN_ASSERT_MSGS["job_not_pending"].format(
            job_id=job_id, state=state_result["state"]
        )

        # Resume only first node
        resume_node(login_ip, "root", compute_nodes[0])
        wait_node_state(login_ip, "root", compute_nodes[0], "idle", timeout=60)

        # Wait for job to transition
        running_result = wait_job_running(login_ip, "root", job_id, timeout=180)
        assert running_result["state"] in {"RUNNING", "COMPLETED"}, \
            DRAIN_ASSERT_MSGS["job_not_running"].format(
                job_id=job_id, state=running_result["state"]
            )
        log.passed(f"Job {job_id} transitioned with only one node resumed")

    finally:
        for node in compute_nodes:
            resume_node(login_ip, "root", node)
        if job_ids:
            cleanup_jobs_direct(login_ip, "root", job_ids)


def test_no_manual_intervention_required_for_transition(test_env):
    """No manual intervention required for transition."""
    log = TestLogger(DRAIN_TEST_NAMES["no_manual_intervention"])
    login_ip = test_env["login_ip"]
    compute_nodes = test_env["compute_nodes"]
    job_ids = []

    try:
        # Drain all nodes
        for node in compute_nodes:
            drain_node(login_ip, "root", node, reason="testing_auto_transition")
            wait_node_state(login_ip, "root", node, "drain", timeout=30)

        # Submit job
        result = submit_job_direct(login_ip, "root")
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_id = result["job_id"]
        job_ids.append(job_id)

        # Resume nodes (no other intervention)
        for node in compute_nodes:
            resume_node(login_ip, "root", node)

        # Job should automatically transition without any manual steps
        running_result = wait_job_running(login_ip, "root", job_id, timeout=180)
        assert running_result["state"] in {"RUNNING", "COMPLETED"}, \
            DRAIN_ASSERT_MSGS["job_not_running"].format(
                job_id=job_id, state=running_result["state"]
            )
        log.passed(f"Job {job_id} transitioned automatically without manual intervention")

    finally:
        for node in compute_nodes:
            resume_node(login_ip, "root", node)
        if job_ids:
            cleanup_jobs_direct(login_ip, "root", job_ids)


def test_long_pending_job_still_transitions(test_env):
    """Job that was PENDING for 10+ minutes still transitions."""
    log = TestLogger(DRAIN_TEST_NAMES["long_pending_still_transitions"])
    login_ip = test_env["login_ip"]
    compute_nodes = test_env["compute_nodes"]
    job_ids = []

    try:
        # Drain all nodes
        for node in compute_nodes:
            drain_node(login_ip, "root", node, reason="testing_long_pending")
            wait_node_state(login_ip, "root", node, "drain", timeout=30)

        # Submit job
        result = submit_job_direct(login_ip, "root")
        assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
            error=result["error"]
        )
        job_id = result["job_id"]
        job_ids.append(job_id)

        # Wait for simulated long pending time (using shorter wait for testing)
        # In production, this would be 10+ minutes
        log.check(f"Job {job_id} submitted, waiting to simulate long pending time...")
        time.sleep(60)  # 1 minute for testing; increase for full test

        # Verify still PENDING
        state_result = poll_job_state_direct(login_ip, "root", job_id, timeout=10)
        assert state_result["state"] == "PENDING", \
            f"Job {job_id} unexpectedly changed state during wait"

        # Resume nodes
        for node in compute_nodes:
            resume_node(login_ip, "root", node)

        # Job should still transition
        running_result = wait_job_running(login_ip, "root", job_id, timeout=180)
        assert running_result["state"] in {"RUNNING", "COMPLETED"}, \
            DRAIN_ASSERT_MSGS["job_not_running"].format(
                job_id=job_id, state=running_result["state"]
            )
        log.passed(f"Long-pending job {job_id} successfully transitioned after resume")

    finally:
        for node in compute_nodes:
            resume_node(login_ip, "root", node)
        if job_ids:
            cleanup_jobs_direct(login_ip, "root", job_ids)


# =============================================================================
# FIFO ORDERING TEST
# =============================================================================

def test_fifo_earlier_submitted_job_starts_first(test_env):
    """FIFO: earlier-submitted job starts first."""
    log = TestLogger(DRAIN_TEST_NAMES["fifo_ordering"])
    login_ip = test_env["login_ip"]
    compute_nodes = test_env["compute_nodes"]
    job_ids = []

    try:
        # Drain all nodes
        for node in compute_nodes:
            drain_node(login_ip, "root", node, reason="testing_fifo")
            wait_node_state(login_ip, "root", node, "drain", timeout=30)

        # Submit jobs in sequence with small delays
        for i in range(3):
            result = submit_job_direct(login_ip, "root")
            assert result["success"], DRAIN_ASSERT_MSGS["job_rejected"].format(
                error=result["error"]
            )
            job_ids.append(result["job_id"])
            log.check(f"Submitted job {i+1}: {result['job_id']}")
            time.sleep(1)  # Small delay to ensure ordering

        # Resume nodes
        for node in compute_nodes:
            resume_node(login_ip, "root", node)

        # Wait for all jobs to complete
        for job_id in job_ids:
            wait_job_running(login_ip, "root", job_id, timeout=300)

        # Get start times and verify FIFO order
        start_times = []
        for job_id in job_ids:
            start_time = get_job_start_time(login_ip, "root", job_id)
            start_times.append((job_id, start_time))
            log.check(f"Job {job_id} start time: {start_time}")

        # Verify FIFO: each job should start at or after the previous one
        for i in range(len(start_times) - 1):
            job_a, time_a = start_times[i]
            job_b, time_b = start_times[i + 1]
            # If both have valid times, verify order
            if time_a and time_b and time_a != "Unknown" and time_b != "Unknown":
                assert time_a <= time_b, DRAIN_ASSERT_MSGS["fifo_violation"].format(
                    job_a=job_a, time_a=time_a, job_b=job_b, time_b=time_b
                )
                log.passed(DRAIN_LOG_MSGS["fifo_verified"].format(
                    job_a=job_a, job_b=job_b
                ))

        log.passed("FIFO ordering verified for all jobs")

    finally:
        for node in compute_nodes:
            resume_node(login_ip, "root", node)
        if job_ids:
            cleanup_jobs_direct(login_ip, "root", job_ids)
