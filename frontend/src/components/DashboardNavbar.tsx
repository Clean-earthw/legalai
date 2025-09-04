'use client'

import Link from 'next/link'
import { useUser,useSession } from '@clerk/nextjs'
import { UserButton } from '@clerk/nextjs'

export default function DashboardNavbar() {
  const { user } = useUser()

  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link href="/dashboard" className="text-xl font-bold text-gray-900">
            AI Chat Dashboard
          </Link>
          
          <div className="flex items-center space-x-4">
            <span className="text-gray-700 text-sm">
              Welcome, {user?.firstName || 'User'}!
            </span>
            <UserButton afterSignOutUrl="/" />
          </div>
        </div>
      </div>
    </nav>
  )
}