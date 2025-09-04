'use client'

import { useUser, useAuth as useClerkAuth } from '@clerk/nextjs'

export function useAuth() {
  const { user, isLoaded, isSignedIn } = useUser()
  const { userId, sessionId, getToken } = useClerkAuth()

  return {
    user,
    userId,
    sessionId,
    isLoaded,
    isSignedIn,
    getToken,
    // Additional helper methods
    isAuthenticated: isLoaded && isSignedIn,
    isLoading: !isLoaded,
  }
}
