import { useAuth } from '@clerk/clerk-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const useSetuAPI = () => {
    const { getToken, isSignedIn } = useAuth()

    const createTask = async (description, priority = 'medium', service = null) => {
        if (!isSignedIn) {
            throw new Error('User must be signed in to create tasks')
        }

        const token = await getToken()

        const body = { description, priority }
        if (service) body.service = service

        const response = await fetch(`${API_BASE_URL}/api/v1/tasks/create`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(body)
        })

        if (!response.ok) {
            throw new Error(`Task creation failed: ${response.statusText}`)
        }

        return response.json()
    }

    const getTask = async (taskId) => {
        if (!isSignedIn) {
            throw new Error('User must be signed in')
        }

        const token = await getToken()

        const response = await fetch(`${API_BASE_URL}/api/v1/tasks/${taskId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })

        if (!response.ok) {
            throw new Error(`Failed to fetch task: ${response.statusText}`)
        }

        return response.json()
    }

    const listTasks = async (status = null) => {
        if (!isSignedIn) {
            throw new Error('User must be signed in')
        }

        const token = await getToken()
        const url = status
            ? `${API_BASE_URL}/api/v1/tasks?status=${status}`
            : `${API_BASE_URL}/api/v1/tasks`

        const response = await fetch(url, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })

        if (!response.ok) {
            throw new Error(`Failed to fetch tasks: ${response.statusText}`)
        }

        return response.json()
    }

    const initiateGoogleAuth = async () => {
        if (!isSignedIn) {
            throw new Error('User must be signed in')
        }

        const token = await getToken()

        const response = await fetch(`${API_BASE_URL}/api/v1/auth/google/login`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })

        const data = await response.json()

        // Redirect to Google OAuth
        if (data.auth_url) {
            window.location.href = data.auth_url
        }

        return data
    }

    const initiateNotionAuth = async () => {
        if (!isSignedIn) {
            throw new Error('User must be signed in')
        }

        const token = await getToken()

        const response = await fetch(`${API_BASE_URL}/api/v1/auth/notion/login`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })

        const data = await response.json()

        if (data.auth_url) {
            window.location.href = data.auth_url
        }

        return data
    }

    const initiateGitHubAuth = async () => {
        if (!isSignedIn) {
            throw new Error('User must be signed in')
        }

        const token = await getToken()

        const response = await fetch(`${API_BASE_URL}/api/v1/auth/github/login`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })

        const data = await response.json()

        if (data.auth_url) {
            window.location.href = data.auth_url
        }

        return data
    }

    const initiateSlackAuth = async () => {
        if (!isSignedIn) {
            throw new Error('User must be signed in')
        }

        const token = await getToken()

        const response = await fetch(`${API_BASE_URL}/api/v1/auth/slack/login`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })

        const data = await response.json()

        if (data.auth_url) {
            window.location.href = data.auth_url
        }

        return data
    }

    const getIntegrations = async () => {
        if (!isSignedIn) {
            return []
        }

        const token = await getToken()

        const response = await fetch(`${API_BASE_URL}/api/v1/integrations`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })

        if (!response.ok) {
            return []
        }

        return response.json()
    }

    return {
        createTask,
        getTask,
        listTasks,
        initiateGoogleAuth,
        initiateNotionAuth,
        initiateGitHubAuth,
        initiateSlackAuth,
        getIntegrations
    }
}
