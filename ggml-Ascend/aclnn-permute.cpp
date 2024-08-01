#include <iostream>
#include <vector>
#include "common.h"
#include "acl/acl.h"
#include "aclnn-permute.h"
#include "aclnnop/aclnn_permute.h"

int aclnn_permute_func(void* selfDataAddr, void* outDataAddr,
    aclnn_shape_t& selfShape, aclnn_shape_t& outShape,
    aclDataType selfDataType, aclDataType outDataType,
    aclnn_shape_t& dimsData, aclrtStream &stream) {
    
    auto ret = 0;

    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclIntArray *dims = nullptr;

    ret = create_acl_tensor(selfShape, selfDataType, &selfDataAddr, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = create_acl_tensor(outShape, outDataType, &outDataAddr, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    dims = aclCreateIntArray(dimsData.data(), dimsData.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnPermuteGetWorkspaceSize(self, dims, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnPermute(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermute failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclDestroyIntArray(dims);

    if(workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return 0;
}